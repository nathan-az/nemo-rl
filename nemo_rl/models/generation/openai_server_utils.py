# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared helpers for the OpenAI-compatible HTTP generation servers.

These utilities are backend-agnostic: they operate on token-ID lists plus a
tokenizer, with no engine calls. They are shared by the vLLM async worker
(``vllm_worker_async.py``) and the TRT-LLM HTTP server (``trtllm_http_server.py``),
which both put a message-based ``/v1/chat/completions`` layer in front of a token
engine for the agentic NeMo-Gym path. SGLang does not use these — it is driven
token-in/token-out via ``generate(input_ids)`` and never re-templates messages,
so it has no retokenization drift to correct.
"""

import json
from collections.abc import Iterable
from typing import Any


def resolve_terminator_token_ids(
    tokenizer: Any,
    generation_config: dict[str, Any] | None = None,
) -> set[int]:
    """Every token ID at which the model may end a turn.

    ``tokenizer.eos_token_id`` is a single int on an HF tokenizer, even for models
    whose turns do not end on it. Gemma-4 is the motivating case: its tokenizer
    reports ``1`` (``<eos>``), but a turn ends on ``106`` (``<turn|>``) or, inside a
    tool loop, on ``50`` (``<|tool_response>``) -- and ``1`` never appears in a
    rendered conversation at all.

    ``generation_config.json``'s ``eos_token_id`` is the authoritative list, because
    it is what the engine actually stops on, so the effective set is its union with
    the tokenizer's. This mirrors the set ``trtllm_http_server.py`` already builds
    for stop-token trimming, under the same reasoning.

    ``config.json``'s ``eos_token_id`` is *not* a substitute. For Gemma-4 it reads
    ``[1, 106]``, omitting ``<|tool_response>``; that is enough to move the splice
    boundary in :func:`replace_prefix_tokens` onto an earlier ``<turn|>`` and
    duplicate a span of tokens without raising.

    Args:
        tokenizer: Any object exposing ``eos_token_id`` (int, list, or None).
        generation_config: Parsed ``generation_config.json``, e.g. from vLLM's
            ``ModelConfig.try_get_generation_config()``. Reading it is the caller's
            job because it can hit disk; pass it once rather than per request.

    Returns:
        The set of terminator token IDs. May be empty if neither source has one.
    """
    terminators: set[int] = set()

    def _add(token_ids: Any) -> None:
        if isinstance(token_ids, bool):
            return
        if isinstance(token_ids, int):
            terminators.add(token_ids)
        elif isinstance(token_ids, Iterable) and not isinstance(
            token_ids, (str, bytes)
        ):
            terminators.update(t for t in token_ids if isinstance(t, int))

    _add(getattr(tokenizer, "eos_token_id", None))
    if generation_config:
        _add(generation_config.get("eos_token_id"))
    return terminators


def normalize_tool_call_arguments(
    messages: list[Any], *, before_index: int | None = None
) -> None:
    """Make OpenAI tool calls renderable by model chat templates, in place.

    The OpenAI wire format carries ``function.arguments`` as a JSON **string**, but
    some chat templates require a mapping. Gemma-4's rejects the string form
    outright with ``TemplateError: tool_calls[].function.arguments must be a JSON
    object (mapping), not a string``, so a multi-turn rollout replaying its own tool
    calls through the template fails on every turn after the first.

    Unparseable arguments become ``{}`` rather than raising: a template that cannot
    render history is a harder failure than one that renders an empty argument list,
    and the engine request itself is unaffected.

    Args:
        messages: Chat messages to normalize in place.
        before_index: Only normalize messages before this index. ``None`` means all.
    """
    for message in messages[:before_index] if before_index is not None else messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", tool_call)
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            function["arguments"] = (
                parsed_arguments if isinstance(parsed_arguments, dict) else {}
            )


def replace_prefix_tokens(
    tokenizer: Any,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
    terminator_ids: Iterable[int] | None = None,
) -> list[int]:
    """This is a subroutine used inside the OpenAI-compatible Chat Completion server.

    This function is for fixing up the chat template-tokenized messages history
    to match the model output tokenization up to the last assistant turn,
    in order to preserve the monotonic tokens property for optimized multi-turn
    training.

    Some environments (namely NeMo-Gym) require an OpenAI compatible server
    endpoint rather than an inference engine handle. This is fine for the most
    part, but it may cause issues when the environment is used as a part of
    training.

    RL training frameworks train models on token IDs, but the OpenAI compatible
    server communicates in what is basically de-tokenized text. When multiple
    model calls are made to the OpenAI compatible server in a single trajectory,
    model generations in previous model calls may be re-tokenized to something
    that is different than what was generated. This is not too big of an issue
    (that we know of) at inference time, but the log probs the model produces
    are different enough for the differently re-tokenized generation result that
    it causes the training to be off policy. Off policy isn't necessarily a bad
    thing in isolation, but this source of off-policyness may cause unexpected
    issues if not properly accounted for. It also mis-aligns the token ID
    sequences across model calls, which feels very strange during training.

    There are real cases where the model output string _does not match_ the chat
    template tokenization of the parsed model output. A concrete example is
    inconsistent whitespace tokens around tool call special tokens.

    TODO When NeMo RL supports training image generation models, we want to
    revisit and possibly update this function. This issue occurs when the model
    generates tokens that are de-tokenized into text or images, and then
    re-tokenized into tokens. So if there is a situation like that with images
    and image tokenization is non-unique, then we will need to uppdate this
    function.

    The splice boundary is located by terminator count, not position: count the
    terminator tokens in template_prefix_token_ids and cut at the N-th terminator
    in template_token_ids. This is robust to chat templates that strip reasoning
    (<think>) blocks from history when the last message is a user turn -- that
    shifts token positions but not the per-message terminator count, so counting
    still finds the same boundary (and reduces to the last terminator of the
    prefix when nothing is stripped).

    A "terminator" is any token that can end a turn, which is not always the
    tokenizer's ``eos_token_id``. Callers that know better should pass
    ``terminator_ids`` from :func:`resolve_terminator_token_ids`; without it this
    falls back to the scalar ``tokenizer.eos_token_id``, which is correct for
    models whose turns really do end on EOS (Qwen3, Nemotron) and silently wrong
    for models like Gemma-4 whose turns end on a distinct control token.

    Args:
        tokenizer: Used for ``eos_token_id`` when ``terminator_ids`` is omitted,
            and to detokenize token IDs in assertion messages.
        model_prefix_token_ids: Tokens the model actually emitted through the end
            of the last assistant turn. Empty on turn 1.
        template_prefix_token_ids: Re-render of history up to and including the
            last assistant turn. Supplies the terminator count.
        template_token_ids: Re-render of the full prompt for this turn.
        terminator_ids: Token IDs that may end a turn. Defaults to
            ``{tokenizer.eos_token_id}``.

    Returns:
        ``template_token_ids`` with its prefix replaced by the tokens the model
        actually generated, so the result is a literal extension of
        ``model_prefix_token_ids``.

    Example (turn-by-turn, concise; eos_token_id = 2):
        Turn 1:
            - prefill_T1 (template prefill) = [11,12,13,40,41]
            - model output = [220,17,2]  # decodes to " 4" + EOS
            - model_prefix_token_ids = prefill_T1 + model output
              => [11,12,13,40,41,220,17,2]

        Turn 2 (template retokenizes prior assistant text differently):
            - template_prefix_token_ids = [11,12,13,40,41,1001,2]  # 1001 decodes to " 4"
            - template_token_ids = [11,12,13,40,41,1001,2,21,22,40,41]

        replace_prefix_tokens keeps the exact prior model tokens up to EOS and
        resumes from the template after that EOS:
            output => [11,12,13,40,41,220,17,2,21,22,40,41]
    """
    if not model_prefix_token_ids:
        return template_token_ids

    if terminator_ids is None:
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, "Tokenizer must have an EOS token ID"
        terminators = {eos_token_id}
    else:
        terminators = set(terminator_ids)
        assert terminators, "terminator_ids must not be empty when provided"

    # The model isn't guaranteed to end on a terminator (e.g. it hit max_tokens);
    # chat templates always add one, so cut the model input to just before it.
    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids[-1] in terminators:
        model_cut_end -= 1

    # Locate the turn boundary by terminator count rather than token position.
    # Qwen3 templates may strip prior reasoning blocks when re-rendering history;
    # counting preserves the original generated reasoning tokens without
    # requiring a customized chat template.
    count_needed = sum(1 for tid in template_prefix_token_ids if tid in terminators)

    # Zero is unreachable by the loop below -- count_seen only ever reaches
    # count_needed after an increment -- so catch it here with a message that says
    # what actually went wrong. The usual cause is a terminator set that does not
    # match the model: Gemma-4's turns end on <turn|> (106) or <|tool_response>
    # (50), so the scalar EOS (1) appears nowhere in a render and the count is 0.
    assert count_needed > 0, (
        "No terminator token found in template_prefix_token_ids, so there is no "
        f"splice boundary to find. Terminators searched for: {sorted(terminators)}.\n"
        "If this model's turns do not end on tokenizer.eos_token_id, pass "
        "terminator_ids=resolve_terminator_token_ids(tokenizer, generation_config).\n"
        f"Template prefix token IDs: {template_prefix_token_ids}\n\n"
        f"Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}"
    )

    count_seen = 0
    template_cut_start = -1
    for pos, tid in enumerate(template_token_ids):
        if tid in terminators:
            count_seen += 1
            if count_seen == count_needed:
                template_cut_start = pos
                break

    assert template_cut_start >= 0, (
        f"Terminator #{count_needed} not found in template_token_ids "
        f"(only found {count_seen} terminator tokens total, "
        f"searching for {sorted(terminators)})!\n"
        f"Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}\n\n"
        f"Template token IDs (everything that was sent to the model endpoint): {template_token_ids}\n\n"
        f"Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}\n\n"
        f"Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"
    )

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )

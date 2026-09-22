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
"""Tests for the shared on-policy prefix splice used by the vLLM/TRT-LLM OpenAI servers."""

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    normalize_tool_call_arguments,
    replace_prefix_tokens,
    resolve_terminator_token_ids,
)


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    """Turn 1 has no prior model output; the full template is returned unchanged."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    """A template prefix with no EOS has no valid splice boundary and must raise."""

    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    """A tokenizer that has no EOS token cannot locate the splice boundary."""

    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_without_tokenizer_uses_explicit_eos():
    """Callers that only hold token ids (the Megatron prompt preparer) pass eos_token_id."""
    result = replace_prefix_tokens(
        tokenizer=None,
        model_prefix_token_ids=[100, 2],
        template_prefix_token_ids=[9, 2],
        template_token_ids=[9, 2, 77, 88],
        eos_token_id=2,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_without_tokenizer_reports_missing_eos_without_decoding():
    """The failure message must not require a tokenizer to decode."""
    with pytest.raises(AssertionError, match="EOS token #1 not found"):
        replace_prefix_tokens(
            tokenizer=None,
            model_prefix_token_ids=[100, 2],
            template_prefix_token_ids=[9, 2],
            template_token_ids=[9, 77],
            eos_token_id=2,
        )


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    """When the prefix contains multiple EOS tokens, the splice cuts at the last one."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_qwen3_think_shift_picks_assistant_eos_not_user_eos():
    """Non-strict-prefix: Qwen3 strips <think> from history when the last message is a
    user turn, so the template's prefix region is shorter and a later user-turn EOS
    lands within the first len(template_prefix) positions. The count-based algorithm
    must cut at the assistant EOS and preserve the intervening user turn.
    """

    class _T:
        eos_token_id = 2

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids)

    tokenizer = _T()
    model_prefix_token_ids = [11, 12, 99, 99, 99, 55, 2]
    template_prefix_token_ids = [11, 12, 88, 88, 88, 56, 2]
    template_token_ids = [11, 12, 56, 2, 70, 71, 2, 40, 41]

    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )

    assert result == [11, 12, 99, 99, 99, 55, 2, 70, 71, 2, 40, 41]
    assert 70 in result and 71 in result


# --- terminator resolution -------------------------------------------------


class _GemmaLikeTokenizer:
    """Gemma-4's shape: a scalar EOS that never appears in a rendered turn."""

    eos_token_id = 1

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def test_resolve_terminator_token_ids_unions_tokenizer_and_generation_config():
    """generation_config.json's list is authoritative; the tokenizer's EOS joins it."""
    assert resolve_terminator_token_ids(
        _GemmaLikeTokenizer(), {"eos_token_id": [1, 106, 50]}
    ) == {1, 106, 50}


def test_resolve_terminator_token_ids_accepts_scalar_generation_config():
    assert resolve_terminator_token_ids(_GemmaLikeTokenizer(), {"eos_token_id": 2}) == {
        1,
        2,
    }


def test_resolve_terminator_token_ids_without_generation_config_is_just_the_tokenizer():
    """Preserves the pre-existing behaviour for callers with nothing else to offer."""
    assert resolve_terminator_token_ids(_GemmaLikeTokenizer()) == {1}
    assert resolve_terminator_token_ids(_GemmaLikeTokenizer(), {}) == {1}
    assert resolve_terminator_token_ids(_GemmaLikeTokenizer(), {"top_p": 0.95}) == {1}


def test_resolve_terminator_token_ids_tolerates_missing_eos():
    class _NoEos:
        eos_token_id = None

    assert resolve_terminator_token_ids(_NoEos()) == set()


# --- the Gemma-4 regression ------------------------------------------------
#
# Token IDs below are the real ones, measured against google/gemma-4-31B-it:
# 1 = <eos>, 50 = <|tool_response>, 106 = <turn|>, 107 = "\n". A turn ends on
# 106, or on 50 inside a tool loop; 1 appears nowhere in a rendered
# conversation. The scenario is the production one -- turn 1 the policy emits a
# tool call and stops, the agent appends the tool response, turn 2 re-renders.

_GEMMA_MODEL_PREFIX = [2, 105, 20, 106, 105, 30, 48, 31, 49, 50]
_GEMMA_TEMPLATE_PREFIX = [2, 105, 20, 106, 105, 30, 48, 31, 49, 50, 107]
_GEMMA_TEMPLATE = [2, 105, 20, 106, 105, 30, 48, 31, 49, 50, 107, 51, 106, 105, 40]


def test_replace_prefix_tokens_gemma4_scalar_eos_raises_with_actionable_message():
    """The defect: Gemma-4's scalar EOS is absent from every render, so the
    terminator count is 0 and there is no boundary to find. Must fail loudly and
    name the fix rather than reporting a confusing "EOS #0 not found".
    """
    with pytest.raises(AssertionError, match="No terminator token found"):
        replace_prefix_tokens(
            tokenizer=_GemmaLikeTokenizer(),
            model_prefix_token_ids=_GEMMA_MODEL_PREFIX,
            template_prefix_token_ids=_GEMMA_TEMPLATE_PREFIX,
            template_token_ids=_GEMMA_TEMPLATE,
        )


def test_replace_prefix_tokens_gemma4_generation_config_terminators_reproduce_template():
    """With generation_config.json's [1, 106, 50] the splice succeeds, and because
    Gemma-4 has no retokenization drift it must reproduce the template exactly.
    """
    result = replace_prefix_tokens(
        tokenizer=_GemmaLikeTokenizer(),
        model_prefix_token_ids=_GEMMA_MODEL_PREFIX,
        template_prefix_token_ids=_GEMMA_TEMPLATE_PREFIX,
        template_token_ids=_GEMMA_TEMPLATE,
        terminator_ids={1, 106, 50},
    )
    assert result == _GEMMA_TEMPLATE
    # Gym chains per-call deltas by hash, so this is the load-bearing property.
    assert result[: len(_GEMMA_MODEL_PREFIX)] == _GEMMA_MODEL_PREFIX


def test_replace_prefix_tokens_gemma4_config_json_terminators_duplicate_tokens():
    """config.json's [1, 106] omits <|tool_response> (50), so the count drops by
    one, the boundary lands on the earlier <turn|>, and tokens are duplicated --
    silently, which is worse than the crash. Pinned so nobody "fixes" the bug by
    reaching for config.json.
    """
    result = replace_prefix_tokens(
        tokenizer=_GemmaLikeTokenizer(),
        model_prefix_token_ids=_GEMMA_MODEL_PREFIX,
        template_prefix_token_ids=_GEMMA_TEMPLATE_PREFIX,
        template_token_ids=_GEMMA_TEMPLATE,
        terminator_ids={1, 106},
    )
    assert result != _GEMMA_TEMPLATE
    assert len(result) > len(_GEMMA_TEMPLATE)


def test_replace_prefix_tokens_trims_model_prefix_ending_on_nonscalar_terminator():
    """The trim-trailing-terminator step must also recognize non-EOS terminators,
    or the spliced output carries a duplicate turn-end token.
    """
    result = replace_prefix_tokens(
        tokenizer=_GemmaLikeTokenizer(),
        model_prefix_token_ids=[2, 105, 30, 106],
        template_prefix_token_ids=[2, 105, 30, 106],
        template_token_ids=[2, 105, 30, 106, 105, 40],
        terminator_ids={1, 106},
    )
    assert result == [2, 105, 30, 106, 105, 40]


def test_replace_prefix_tokens_empty_terminator_ids_raises():
    with pytest.raises(AssertionError, match="must not be empty"):
        replace_prefix_tokens(
            tokenizer=_GemmaLikeTokenizer(),
            model_prefix_token_ids=[1, 2],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2, 3],
            terminator_ids=set(),
        )


# --- tool-call argument normalization --------------------------------------


def test_normalize_tool_call_arguments_parses_the_openai_string_form():
    """Gemma-4's template rejects a JSON string outright, so the wire form has to
    be deserialized before rendering history.
    """
    messages = [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"q": "onboarding"}'}}
            ],
        },
    ]
    normalize_tool_call_arguments(messages)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"q": "onboarding"}


def test_normalize_tool_call_arguments_leaves_mappings_and_others_alone():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "s", "arguments": {"q": "x"}}}],
        },
        {"role": "user", "content": '{"not": "a tool call"}'},
        {"role": "assistant", "content": "no tool calls here"},
    ]
    normalize_tool_call_arguments(messages)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert messages[1]["content"] == '{"not": "a tool call"}'


def test_normalize_tool_call_arguments_unparseable_becomes_empty_mapping():
    """A template that renders empty arguments beats one that cannot render at all."""
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"arguments": "not json"}}]},
        {"role": "assistant", "tool_calls": [{"function": {"arguments": "[1, 2]"}}]},
    ]
    normalize_tool_call_arguments(messages)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {}


def test_normalize_tool_call_arguments_respects_before_index():
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"a": 1}'}}]},
        {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"b": 2}'}}]},
    ]
    normalize_tool_call_arguments(messages, before_index=1)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"b": 2}'

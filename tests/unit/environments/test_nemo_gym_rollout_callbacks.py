# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""env.nemo_gym.rollout_result_callbacks: config, payload, isolation, plumbing.

The callbacks are an observability seam: each completed Gym rollout is handed
to user code with Gym's raw result, before NeMo-RL postprocesses (and may
penalise or zero) the reward. Nothing a callback does may break training, and a
run that configures none must not pay for the feature.
"""

import asyncio
import logging
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest
from omegaconf import OmegaConf

import nemo_rl.environments.nemo_gym as nemo_gym_mod
import nemo_rl.environments.nemo_gym_rollout_callbacks as callbacks_mod
from nemo_rl.environments.nemo_gym import NemoGym, build_nemo_gym_config
from nemo_rl.environments.nemo_gym_rollout_callbacks import (
    RolloutCallbackContext,
    RolloutResultCallbacks,
    RolloutResultEvent,
    resolve_rollout_result_callback,
    validate_rollout_result_callback_paths,
)
from nemo_rl.experience.rollouts import _merge_nemo_gym_instance_streams

# NemoGym is a Ray actor; grab the plain class so these run without a cluster.
NemoGymClass = NemoGym.__ray_metadata__.modified_class

_MODULE = "fake_rollout_callbacks"


@pytest.fixture
def callback_module(monkeypatch):
    """An importable module of callbacks that record what they receive."""
    module = types.ModuleType(_MODULE)
    module.calls = []

    def first(event):
        module.calls.append(("first", event))

    def second(event):
        module.calls.append(("second", event))

    def boom(event):
        module.calls.append(("boom", event))
        raise RuntimeError("tracker is down")

    def mutate(event):
        event.result["reward"] = 0.0

    async def async_callback(event):
        module.calls.append(("async", event))

    class Tracker:
        def __init__(self):
            self.events = []

        def on_rollout(self, event):
            self.events.append(event)

    module.first = first
    module.second = second
    module.boom = boom
    module.mutate = mutate
    module.async_callback = async_callback
    module.tracker = Tracker()
    module.not_callable = 42
    monkeypatch.setitem(sys.modules, _MODULE, module)
    return module


def _row(**extra: Any) -> dict:
    row = {
        "_rowidx": 0,
        "agent_ref": {"type": "responses_api_agents", "name": "math_agent"},
        "_ng_task_index": 7,
        "_ng_group_id": "group-a",
        "_ng_group_attempt": 1,
        "_ng_rollout_index": 3,
    }
    row.update(extra)
    return row


class TestConfigValidation:
    def test_accepts_a_list_and_an_omegaconf_list(self):
        assert validate_rollout_result_callback_paths(["a.b:c", "d:e.f"]) == [
            "a.b:c",
            "d:e.f",
        ]
        listconfig = OmegaConf.create({"cbs": ["a.b:c"]})["cbs"]
        assert validate_rollout_result_callback_paths(listconfig) == ["a.b:c"]

    @pytest.mark.parametrize(
        "paths",
        ["a.b:c", [1], {"a": "b:c"}, None],
        ids=["bare-string", "non-string-entry", "mapping", "none"],
    )
    def test_rejects_anything_but_a_list_of_strings(self, paths):
        with pytest.raises(TypeError, match="list of 'module:callable'"):
            validate_rollout_result_callback_paths(paths)

    @pytest.mark.parametrize("path", ["a.b.c", ":c", "a.b:", ""])
    def test_rejects_entries_that_are_not_module_colon_attribute(self, path):
        with pytest.raises(ValueError, match="'module:callable' import paths"):
            validate_rollout_result_callback_paths([path])

    def test_rejects_a_duplicate_entry(self):
        with pytest.raises(ValueError, match="twice"):
            validate_rollout_result_callback_paths(["a:b", "a:b"])

    def test_build_config_moves_callbacks_to_an_actor_field(self, monkeypatch):
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)
        cfg = build_nemo_gym_config(
            {
                "nemo_gym": {
                    "config_paths": ["gym.yaml"],
                    "rollout_result_callbacks": ["my_pkg.hooks:on_rollout"],
                }
            },
            base_urls=["http://vllm-0"],
            model_name="m",
            enable_router_replay=False,
            use_fastokens=False,
        )
        assert cfg["rollout_result_callbacks"] == ["my_pkg.hooks:on_rollout"]
        # A NeMo-RL key: it must never reach Gym's global config.
        assert cfg["initial_global_config_dict"] == {"config_paths": ["gym.yaml"]}

    @pytest.mark.parametrize("configured", [None, []], ids=["null", "empty"])
    def test_build_config_without_callbacks_adds_no_field(
        self, monkeypatch, configured
    ):
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)
        cfg = build_nemo_gym_config(
            {
                "nemo_gym": {
                    "config_paths": ["gym.yaml"],
                    "rollout_result_callbacks": configured,
                }
            },
            base_urls=[],
            model_name="m",
            enable_router_replay=False,
            use_fastokens=False,
        )
        assert "rollout_result_callbacks" not in cfg
        assert cfg["initial_global_config_dict"] == {"config_paths": ["gym.yaml"]}

    def test_build_config_rejects_a_malformed_entry_on_the_driver(self, monkeypatch):
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
        monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)
        with pytest.raises(ValueError, match="'module:callable' import paths"):
            build_nemo_gym_config(
                {"nemo_gym": {"rollout_result_callbacks": ["no_colon_here"]}},
                base_urls=[],
                model_name="m",
                enable_router_replay=False,
                use_fastokens=False,
            )


class TestResolution:
    def test_resolves_a_module_attribute(self, callback_module):
        assert resolve_rollout_result_callback(f"{_MODULE}:first") is (
            callback_module.first
        )

    def test_resolves_a_dotted_attribute(self, callback_module):
        resolved = resolve_rollout_result_callback(f"{_MODULE}:tracker.on_rollout")
        assert resolved == callback_module.tracker.on_rollout

    def test_an_unimportable_module_fails(self):
        with pytest.raises(ModuleNotFoundError):
            resolve_rollout_result_callback("no_such_module_for_this_test:fn")

    def test_a_missing_attribute_fails(self, callback_module):
        with pytest.raises(AttributeError):
            resolve_rollout_result_callback(f"{_MODULE}:missing")

    def test_a_non_callable_fails(self, callback_module):
        with pytest.raises(TypeError, match="not callable"):
            resolve_rollout_result_callback(f"{_MODULE}:not_callable")

    def test_a_coroutine_function_fails_instead_of_never_running(self, callback_module):
        with pytest.raises(TypeError, match="coroutine function"):
            resolve_rollout_result_callback(f"{_MODULE}:async_callback")

    @pytest.mark.parametrize("configured", [None, []], ids=["absent", "empty"])
    def test_no_callbacks_configured_builds_nothing(self, configured):
        assert RolloutResultCallbacks.from_config(configured) is None


class TestInvocation:
    def test_callbacks_run_in_configured_order_with_one_shared_event(
        self, callback_module
    ):
        callbacks = RolloutResultCallbacks.from_config(
            [f"{_MODULE}:second", f"{_MODULE}:first"]
        )
        row = _row(_ng_rollout_id="rollout-xyz")
        result = {"reward": 1, "response": {"output": []}}
        context = RolloutCallbackContext(step=12, weight_version=11, phase="train")

        callbacks(row, result, context)

        assert [name for name, _ in callback_module.calls] == ["second", "first"]
        event = callback_module.calls[0][1]
        assert callback_module.calls[1][1] is event
        assert isinstance(event, RolloutResultEvent)
        assert event.reward == 1.0
        assert event.agent_name == "math_agent"
        assert event.rollout_id == "rollout-xyz"
        assert event.task_index == 7
        assert event.group_id == "group-a"
        assert event.group_attempt == 1
        assert event.rollout_index == 3
        assert event.context == context
        assert event.row == row
        assert event.result == result

    def test_identity_fields_are_none_when_gym_did_not_stamp_them(
        self, callback_module
    ):
        callbacks = RolloutResultCallbacks.from_config([f"{_MODULE}:first"])
        callbacks({"_rowidx": 0}, {"response": {}}, None)
        event = callback_module.calls[0][1]
        assert event.reward is None
        assert event.agent_name is None
        assert event.rollout_id is None
        assert event.task_index is None
        assert event.context is None

    def test_a_non_numeric_reward_is_reported_as_none_rather_than_raising(
        self, callback_module
    ):
        callbacks = RolloutResultCallbacks.from_config([f"{_MODULE}:first"])
        callbacks(_row(), {"reward": "n/a"}, None)
        assert callback_module.calls[0][1].reward is None

    def test_views_are_read_only_and_not_copies(self, callback_module):
        callbacks = RolloutResultCallbacks.from_config(
            [f"{_MODULE}:mutate", f"{_MODULE}:first"]
        )
        result = {"reward": 1.0, "response": {"output": []}}
        callbacks(_row(), result, None)

        # mutate's top-level write was refused (and isolated) ...
        assert result["reward"] == 1.0
        assert callbacks.failure_counts[f"{_MODULE}:mutate"] == 1
        # ... and the view shares the nested objects rather than copying them.
        event = callback_module.calls[0][1]
        assert event.result["response"] is result["response"]
        with pytest.raises(TypeError):
            event.row["agent_ref"] = {}  # type: ignore[index]


class TestFailureIsolation:
    def test_a_raising_callback_does_not_stop_the_next_one(self, callback_module):
        callbacks = RolloutResultCallbacks.from_config(
            [f"{_MODULE}:boom", f"{_MODULE}:first"]
        )
        callbacks(_row(), {"reward": 0.5}, None)
        callbacks(_row(), {"reward": 0.5}, None)

        assert [name for name, _ in callback_module.calls] == [
            "boom",
            "first",
            "boom",
            "first",
        ]
        assert callbacks.failure_counts == {
            f"{_MODULE}:boom": 2,
            f"{_MODULE}:first": 0,
        }

    def test_failures_are_logged_once_per_interval_per_callback(
        self, callback_module, monkeypatch, caplog
    ):
        now = [1000.0]
        monkeypatch.setattr(callbacks_mod, "monotonic", lambda: now[0])
        callbacks = RolloutResultCallbacks.from_config([f"{_MODULE}:boom"])

        def failure_logs():
            return [
                record
                for record in caplog.records
                if record.name == callbacks_mod.__name__
            ]

        with caplog.at_level(logging.ERROR, logger=callbacks_mod.__name__):
            for _ in range(5):
                callbacks(_row(), {"reward": 0.0}, None)
            assert len(failure_logs()) == 1
            assert failure_logs()[0].exc_info is not None
            assert "tracker is down" in caplog.text

            now[0] += callbacks_mod.CALLBACK_FAILURE_LOG_INTERVAL_S
            callbacks(_row(), {"reward": 0.0}, None)

        assert len(failure_logs()) == 2
        assert "6 failure(s) so far" in failure_logs()[1].getMessage()
        assert callbacks.failure_counts[f"{_MODULE}:boom"] == 6

    def test_base_exceptions_are_not_swallowed(self, callback_module, monkeypatch):
        def cancelled(event):
            raise asyncio.CancelledError

        monkeypatch.setattr(callback_module, "cancelled", cancelled, raising=False)
        callbacks = RolloutResultCallbacks.from_config([f"{_MODULE}:cancelled"])
        with pytest.raises(asyncio.CancelledError):
            callbacks(_row(), {"reward": 0.0}, None)


class _RolloutHelper:
    """Gym's RolloutCollectionHelper: one awaitable (row, result) per example."""

    def __init__(self, rewards: list[float]):
        self.rewards = rewards

    def run_examples(self, examples: list[dict], head_server_config: str) -> list:
        async def completed(example, reward):
            return example, {"reward": reward, "response": {"output": []}}

        return [
            completed(example, reward)
            for example, reward in zip(examples, self.rewards)
        ]


def _actor(callbacks: list[str] | None) -> Any:
    cfg = {"model_name": "m", "base_urls": [], "initial_global_config_dict": {}}
    if callbacks is not None:
        cfg["rollout_result_callbacks"] = callbacks
    env = NemoGymClass(cfg)
    env.rh = object()
    env._tokenizer = object()
    env.head_server_config = "head-server"
    env.rch = _RolloutHelper([1.0, 0.25])
    return env


def _penalising_postprocess(seen_by_postprocess: list):
    """Stand-in for postprocessing that rewrites the reward in place."""

    def postprocess(row, result, tokenizer, **kwargs):
        seen_by_postprocess.append(result["reward"])
        result["reward"] = 0.0
        return {"message_log": [], "full_result": result}

    return postprocess


async def _collect(stream: AsyncIterator[Any]) -> list:
    return [item async for item in stream]


def _rows() -> list[dict]:
    return [_row(_rowidx=0, _ng_rollout_index=0), _row(_rowidx=1, _ng_rollout_index=1)]


class TestActor:
    def test_callbacks_see_the_raw_reward_before_postprocessing(self, callback_module):
        env = _actor([f"{_MODULE}:first"])
        seen_by_postprocess: list[float] = []
        env._postprocess_nemo_gym_to_nemo_rl_result = _penalising_postprocess(
            seen_by_postprocess
        )
        context = RolloutCallbackContext(step=3, phase="validation")

        streamed = asyncio.run(
            _collect(env.run_rollouts(_rows(), "timing/test", rollout_context=context))
        )

        assert [event.reward for _, event in callback_module.calls] == [1.0, 0.25]
        assert [event.rollout_index for _, event in callback_module.calls] == [0, 1]
        assert all(event.context is context for _, event in callback_module.calls)
        # Postprocessing still ran, after the callback, on the same raw values.
        assert seen_by_postprocess == [1.0, 0.25]
        assert [item[2]["full_result"]["reward"] for item in streamed] == [0.0, 0.0]
        assert "timing/test/rollout_result_callbacks" in streamed[-1][3]

    def test_a_raising_callback_does_not_break_the_stream(self, callback_module):
        env = _actor([f"{_MODULE}:boom", f"{_MODULE}:first"])
        env._postprocess_nemo_gym_to_nemo_rl_result = _penalising_postprocess([])

        streamed = asyncio.run(_collect(env.run_rollouts(_rows(), "timing/test")))

        assert [item[0] for item in streamed] == [0, 1]
        assert env._rollout_result_callbacks.failure_counts[f"{_MODULE}:boom"] == 2
        assert [name for name, _ in callback_module.calls].count("first") == 2

    def test_no_callbacks_is_a_no_op(self):
        env = _actor(None)
        env._postprocess_nemo_gym_to_nemo_rl_result = _penalising_postprocess([])
        assert env._rollout_result_callbacks is None

        streamed = asyncio.run(_collect(env.run_rollouts(_rows(), "timing/test")))

        assert [item[0] for item in streamed] == [0, 1]
        assert not any("rollout_result_callbacks" in key for key in streamed[-1][3])

    def test_an_unimportable_callback_fails_actor_construction(self):
        with pytest.raises(ModuleNotFoundError):
            _actor(["no_such_module_for_this_test:fn"])


class _FakeRunRollouts:
    def __init__(self, calls: list):
        self.calls = calls

    def options(self, **kwargs):
        return self

    def remote(self, rows, timer_prefix, deduplicate_multimodal_data, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            for row in rows:

                async def item(row=row):
                    return row["_rowidx"], row["agent_ref"], {}, None

                yield item()

        return stream()


class _FakeActorHandle:
    def __init__(self):
        self.calls: list = []
        self.run_rollouts = _FakeRunRollouts(self.calls)


class TestPlumbing:
    def _drain(self, rollout_context):
        handle = _FakeActorHandle()

        async def run():
            return [
                item
                async for item in _merge_nemo_gym_instance_streams(
                    [("gym", handle, [_row(_rowidx=0)])],
                    "timing/test",
                    rollout_context=rollout_context,
                )
            ]

        assert len(asyncio.run(run())) == 1
        return handle.calls

    def test_the_context_reaches_the_actor_call(self):
        context = RolloutCallbackContext(step=5, weight_version=4, phase="train")
        assert self._drain(context) == [{"rollout_context": context}]

    def test_no_context_keeps_the_original_call_signature(self):
        assert self._drain(None) == [{}]

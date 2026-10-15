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
"""Per-rollout result callbacks for the NeMo-Gym environment.

``env.nemo_gym.rollout_result_callbacks`` lists ``"module:callable"`` import
paths. The ``NemoGym`` actor resolves them once at construction and calls each
one with a :class:`RolloutResultEvent` for every rollout Gym completes, right
after the raw ``(row, result)`` pair arrives and before NeMo-RL postprocesses
it -- that is, before reward penalties, length penalties or masking can rewrite
the reward. It is a vendor-neutral seam for observability tools (experiment
trackers, rollout databases) that need each rollout's raw result.

Callbacks run synchronously on the ``NemoGym`` actor's event loop, inside that
actor's process and Python environment, so they must be importable there and
must return quickly: do any I/O on a thread or queue of your own. An exception
raised by a callback is caught, counted and logged (rate limited); it never
fails the rollout or the training step.
"""

import importlib
import inspect
import logging
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal, Optional, cast

from nemo_rl.experience.interfaces import (
    NEMO_GYM_GROUP_ATTEMPT_KEY,
    NEMO_GYM_GROUP_ID_KEY,
    NEMO_GYM_ROLLOUT_INDEX_KEY,
    NEMO_GYM_TASK_INDEX_KEY,
)

logger = logging.getLogger(__name__)

# Gym's ROLLOUT_ID_KEY_NAME; stamped on rows only when token capture is enabled.
NEMO_GYM_ROLLOUT_ID_KEY = "_ng_rollout_id"

# After the first failure of a callback (always logged with its traceback),
# further failures of the same callback are logged at most this often.
CALLBACK_FAILURE_LOG_INTERVAL_S = 60.0

RolloutPhase = Literal["train", "validation"]


@dataclass(frozen=True)
class RolloutCallbackContext:
    """Training context for one ``NemoGym.run_rollouts`` batch.

    Supplied by the caller that starts the batch, since only the training loop
    knows it. Every field is optional: a caller that does not know a value
    leaves it ``None`` rather than guessing.

    Attributes:
        step: Training step the batch is collected for.
        weight_version: Policy weight version generation started from. Differs
            from ``step`` under async GRPO, where rollouts may lag the trainer.
        phase: ``"train"`` or ``"validation"``.
    """

    # Defaults keep existing constructors valid when a field is added later.
    step: Optional[int] = None
    weight_version: Optional[int] = None
    phase: Optional[RolloutPhase] = None


@dataclass(frozen=True)
class RolloutResultEvent:
    """What a rollout result callback receives, once per completed rollout.

    ``row`` and ``result`` are read-only views of the objects NeMo-Gym
    returned. The views are shallow: nested values are the live objects
    NeMo-RL postprocesses next, so a callback must not mutate them and must
    copy anything it keeps beyond the call or hands to another thread.

    Attributes:
        row: The Gym input row as Gym echoed it back (``agent_ref`` resolved).
        result: Gym's raw rollout result, including its unmodified ``reward``.
        reward: ``result["reward"]`` as Gym reported it, or ``None`` if absent
            or not a number.
        agent_name: ``row["agent_ref"]["name"]``, or ``None`` if absent.
        rollout_id: ``_ng_rollout_id`` when token capture stamped one.
        task_index: ``_ng_task_index`` (async GRPO prompt ordinal), if stamped.
        group_id: ``_ng_group_id`` of the prompt group, if stamped.
        group_attempt: ``_ng_group_attempt`` (retry count of the group), if stamped.
        rollout_index: ``_ng_rollout_index`` (generation index in the group), if stamped.
        context: Training context the caller passed to ``run_rollouts``.
    """

    row: Mapping[str, Any]
    result: Mapping[str, Any]
    reward: Optional[float]
    agent_name: Optional[str]
    rollout_id: Optional[str]
    task_index: Optional[int]
    group_id: Optional[str]
    group_attempt: Optional[int]
    rollout_index: Optional[int]
    context: Optional[RolloutCallbackContext]


RolloutResultCallback = Callable[[RolloutResultEvent], None]


def validate_rollout_result_callback_paths(paths: Any) -> list[str]:
    """Check ``rollout_result_callbacks`` entries and return them as a plain list.

    Nothing is imported. This runs on the driver while the actor config is
    built, so a malformed entry fails at setup; importing is left to the actor,
    whose Python environment the callbacks are resolved in.

    Raises:
        TypeError: ``paths`` is not a sequence of strings.
        ValueError: An entry is not of the form ``"module:attribute"``, or
            appears twice (it would be called twice per rollout).
    """
    # Sequence rather than list: an OmegaConf ListConfig must pass too.
    if (
        isinstance(paths, str)
        or not isinstance(paths, Sequence)
        or not all(isinstance(p, str) for p in paths)
    ):
        raise TypeError(
            "env.nemo_gym.rollout_result_callbacks must be a list of "
            f"'module:callable' strings, got {paths!r}"
        )
    for path in paths:
        module_name, sep, attribute = path.partition(":")
        if not sep or not module_name or not attribute:
            raise ValueError(
                "env.nemo_gym.rollout_result_callbacks entries must be "
                f"'module:callable' import paths, got {path!r}"
            )
    if len(set(paths)) != len(paths):
        raise ValueError(
            f"env.nemo_gym.rollout_result_callbacks lists an entry twice: {paths!r}"
        )
    return list(paths)


def resolve_rollout_result_callback(path: str) -> RolloutResultCallback:
    """Import one ``"module:attribute"`` path and return the callable it names.

    ``attribute`` may be dotted (``"pkg.mod:Tracker.instance.on_rollout"``).

    Raises:
        ValueError: ``path`` is not of the form ``"module:attribute"``.
        ImportError: The module cannot be imported.
        AttributeError: The module has no such attribute.
        TypeError: The target is not callable, or is a coroutine function
            (callbacks run synchronously and would never be awaited).
    """
    validate_rollout_result_callback_paths([path])
    module_name, _, attribute = path.partition(":")
    target: Any = importlib.import_module(module_name)
    for name in attribute.split("."):
        target = getattr(target, name)
    if not callable(target):
        raise TypeError(f"rollout result callback {path!r} is not callable")
    if inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        getattr(target, "__call__", None)
    ):
        raise TypeError(
            f"rollout result callback {path!r} is a coroutine function; callbacks "
            "are called synchronously, so it would never run. Use a plain function "
            "that hands work to a thread or queue of its own."
        )
    return cast(RolloutResultCallback, target)


def build_rollout_result_event(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    context: Optional[RolloutCallbackContext],
) -> RolloutResultEvent:
    """Build the event for one raw Gym ``(row, result)`` pair, copying nothing."""
    agent_ref = row.get("agent_ref")
    agent_name = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
    reward = result.get("reward")
    return RolloutResultEvent(
        row=types.MappingProxyType(row),
        result=types.MappingProxyType(result),
        # No float() on arbitrary values: building the event runs outside the
        # per-callback isolation, so it must not be able to raise.
        reward=float(reward) if isinstance(reward, (int, float)) else None,
        agent_name=agent_name,
        rollout_id=row.get(NEMO_GYM_ROLLOUT_ID_KEY),
        task_index=row.get(NEMO_GYM_TASK_INDEX_KEY),
        group_id=row.get(NEMO_GYM_GROUP_ID_KEY),
        group_attempt=row.get(NEMO_GYM_GROUP_ATTEMPT_KEY),
        rollout_index=row.get(NEMO_GYM_ROLLOUT_INDEX_KEY),
        context=context,
    )


class RolloutResultCallbacks:
    """Resolved callbacks plus the failure isolation around calling them.

    Built only when at least one callback is configured, so a run without
    callbacks never constructs events or enters this class.
    """

    def __init__(self, paths: list[str]):
        if not paths:
            raise ValueError("RolloutResultCallbacks needs at least one callback")
        self._paths = list(paths)
        self._callbacks = [resolve_rollout_result_callback(p) for p in self._paths]
        self.failure_counts: dict[str, int] = {path: 0 for path in self._paths}
        self._last_failure_log: dict[str, float] = {}

    @classmethod
    def from_config(
        cls, paths: Optional[Sequence[str]]
    ) -> Optional["RolloutResultCallbacks"]:
        """Return the resolved callbacks, or ``None`` when none are configured."""
        if not paths:
            return None
        return cls(validate_rollout_result_callback_paths(paths))

    def __call__(
        self,
        row: Mapping[str, Any],
        result: Mapping[str, Any],
        context: Optional[RolloutCallbackContext],
    ) -> None:
        """Call every callback, in configuration order, with one event.

        Never raises ``Exception``: a failing callback is counted and logged,
        and the remaining callbacks still run.
        """
        event = build_rollout_result_event(row, result, context)
        for path, callback in zip(self._paths, self._callbacks):
            try:
                callback(event)
            # Deliberately broad: callbacks are user code, and whatever they
            # raise must not fail training. BaseException (cancellation,
            # KeyboardInterrupt) still propagates.
            except Exception:
                self._record_failure(path)

    def _record_failure(self, path: str) -> None:
        self.failure_counts[path] += 1
        count = self.failure_counts[path]
        now = monotonic()
        last = self._last_failure_log.get(path)
        if last is not None and now - last < CALLBACK_FAILURE_LOG_INTERVAL_S:
            return
        self._last_failure_log[path] = now
        logger.exception(
            "NeMo-Gym rollout result callback %r raised (%d failure(s) so far); "
            "the rollout is unaffected. Further failures are logged at most "
            "every %.0fs.",
            path,
            count,
            CALLBACK_FAILURE_LOG_INTERVAL_S,
        )

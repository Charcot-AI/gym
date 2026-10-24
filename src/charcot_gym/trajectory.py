"""Deterministic JSONL trajectories and action-based episode replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict

from charcot_gym.task import Task
from charcot_gym.tools import CORE_TOOLS, ToolRegistry
from charcot_gym.verifier import VERIFIERS, Check, Score, VerifierRegistry


class TrajectoryEntry(BaseModel):
    """One action, resulting observation digest, and step metadata."""

    model_config = ConfigDict(frozen=True)

    action: dict[str, Any]
    observation_hash: str
    info: dict[str, Any]


class Trajectory:
    """An ordered list of replayable environment actions and audit records."""

    def __init__(self, entries: Iterable[TrajectoryEntry] = ()) -> None:
        self.entries = deepcopy(list(entries))

    def append(
        self, action: dict[str, Any], observation: dict[str, Any], info: dict[str, Any]
    ) -> None:
        """Append a step and hash its canonical observation."""
        digest = hashlib.sha256(_canonical_json(observation).encode()).hexdigest()
        self.entries.append(
            TrajectoryEntry(
                action=json.loads(json.dumps(action)),
                observation_hash=digest,
                info=json.loads(json.dumps(info)),
            )
        )

    def to_jsonl(self) -> str:
        """Serialize entries as stable, newline-delimited JSON."""
        return "".join(
            json.dumps(entry.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
            for entry in self.entries
        )

    @classmethod
    def from_jsonl(cls, content: str) -> Trajectory:
        """Parse and validate a newline-delimited trajectory."""
        entries = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(TrajectoryEntry.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid trajectory line {line_number}") from error
        return cls(entries)

    def actions(self) -> list[dict[str, Any]]:
        """Return detached action records for replay."""
        return [json.loads(json.dumps(entry.action)) for entry in self.entries]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def replay(
    task: Task,
    seed: int,
    trajectory: Trajectory | str,
    *,
    verifiers: VerifierRegistry | None = None,
    tools: ToolRegistry | None = None,
) -> Score:
    """Replay actions from a fresh reset and return the resulting score."""
    from charcot_gym.env import ClinicalEnv  # noqa: PLC0415

    recorded = Trajectory.from_jsonl(trajectory) if isinstance(trajectory, str) else trajectory
    environment = ClinicalEnv(
        [task],
        verifiers=verifiers or VERIFIERS,
        tools=tools or CORE_TOOLS,
    )
    environment.reset(task.id, seed=seed)
    for action in recorded.actions():
        environment.step(_decode_action(action))
        if environment.done:
            break
    if not environment.done:
        return Score(
            value=0.0,
            passed=False,
            checks=[
                Check(
                    name="episode_ended",
                    passed=False,
                    detail="The recorded actions did not end the episode.",
                )
            ],
        )
    return environment.score()


def _decode_action(action: dict[str, Any]) -> Any:
    from charcot_gym.env import Submit, ToolCall  # noqa: PLC0415

    if action.get("type") == "tool":
        return ToolCall(name=action["name"], arguments=action["arguments"])
    if action.get("type") == "submit":
        return Submit(answer=action.get("answer"))
    return action

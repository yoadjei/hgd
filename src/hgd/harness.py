"""ReAct harness — audit task T1.1 and section 20.

One loop, one system prompt, one tool schema, zero retries. No planner, no
memory module, no verifier: audit section 20 is explicit that those are
interventions or ablations and must never be part of the baseline, because a
scaffolded baseline would confound every mechanism estimate the paper makes.

The loop's real obligation is to the log. Every step must carry a complete
schema record, and the trajectory must replay bit-exactly from those records
alone — that is the Phase 1 gate and, through kill condition C, the precondition
for any causal claim at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

from hgd.checkpoints import Checkpoint, evaluate_checkpoints
from hgd.env import Environment
from hgd.model import Model
from hgd.parsing import ActionFormat, parse_action
from hgd.schema import StepRecord

# a code block may touch several apis; the leading call is logged as the step's
# nominal tool and tier-1 rules plus state checks handle anything finer
_LEADING_CALL = re.compile(r"([A-Za-z_][\w.]*)\s*\(")


@dataclass(frozen=True)
class HarnessConfig:
    action_format: ActionFormat = ActionFormat.CODE
    step_limit_multiplier: int = 3
    terminal_actions: tuple[str, ...] = ()
    system_prompt: str = ""
    tools: tuple[Mapping[str, Any], ...] = ()
    observation_truncation: int = 4096
    # tier-2 state assertions, evaluated after every step. these cannot be
    # recovered from the log afterwards the way event and judge labels can: a
    # checkpoint reads live environment state, and the world is closed by the time
    # anything reads the jsonl. empty by default because each pass costs an extra
    # evaluate(), which on appworld runs the task's unit tests.
    checkpoints: tuple[Checkpoint, ...] = ()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_run_id(task_id: str, seed: int, model_name: str, branch: str) -> str:
    """Deterministic run identity, so a rerun of the same cell is traceable."""
    payload = json.dumps([task_id, seed, model_name, branch], sort_keys=True)
    return _sha256(payload)[:16]


def _tool_name_of(action: str | None, parsed_tool_name: str | None) -> str | None:
    if parsed_tool_name is not None:
        return parsed_tool_name
    if not action:
        return None
    match = _LEADING_CALL.search(action)
    return match.group(1) if match else None


def run_episode(
    *,
    task_id: str,
    model: Model,
    env: Environment,
    seed: int,
    h_star: int,
    config: HarnessConfig,
    intervention_branch: str = "factual",
    parent_run_id: str | None = None,
    branch_step: int | None = None,
) -> list[StepRecord]:
    """Run one episode and return its complete log.

    ``h_star`` is the intrinsic horizon (HORIZON's definition: minimum effective
    actions under an optimal policy). It sets both the step limit and the
    normalized position ``u`` that the hazard model is indexed by.
    """
    if h_star <= 0:
        raise ValueError("h_star must be positive: u = step / h_star")

    env.reset(task_id, seed)
    # appworld's completion is not idempotent: measured on 2026-09-30, a second
    # apis.supervisor.complete_task() on an already-complete task flipped one of the
    # task's two unit tests and took pass fraction from 1.0 to 0.5. an episode that
    # starts complete would spend its first action destroying P_obs, and the way to
    # get here is to branch from a replayed prefix that already finished. loud,
    # because an empty log returned quietly reads as a model that said nothing.
    if _environment_reports_completion(env):
        raise ValueError(
            f"task {task_id} already reports completion before the first action; "
            "branching from a completed prefix would score the run down rather than "
            "measure it"
        )
    run_id = _make_run_id(task_id, seed, model.name, intervention_branch)
    step_limit = config.step_limit_multiplier * h_star

    messages: list[dict[str, Any]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})

    records: list[StepRecord] = []

    for step in range(step_limit):
        prompt_hash = _sha256(json.dumps(messages, sort_keys=True, default=str))

        started = time.perf_counter()
        response = model.generate(messages, tools=config.tools or None)
        latency_s = time.perf_counter() - started

        parsed = parse_action(
            response.text, config.action_format, list(response.tool_calls) or None
        )

        # two termination paths that differ in whether the action runs. a
        # sentinel like done() must not execute: DictEnvironment would answer
        # "unknown operation" and stamp a tool error on every clean run's last
        # step. appworld's complete_task() must execute, because that call is how
        # the environment learns the episode is over, so it is checked afterwards.
        is_sentinel = (
            parsed.action is not None and parsed.action in config.terminal_actions
        )

        tool_result: str | None = None
        if parsed.action is not None and not is_sentinel:
            tool_result = env.execute(parsed.action)[: config.observation_truncation]

        records.append(
            StepRecord(
                run_id=run_id,
                task_id=task_id,
                seed=seed,
                model=model.name,
                step=step,
                u=step / h_star,
                prompt_hash=prompt_hash,
                raw_output=response.text,
                parsed_action=parsed.action,
                tool_name=_tool_name_of(parsed.action, parsed.tool_name),
                tool_args=dict(parsed.tool_args) if parsed.tool_args else None,
                tool_result=tool_result,
                tool_result_hash=_sha256(tool_result or ""),
                env_state_hash=env.state_hash(),
                # dicts, not CheckpointResult objects: a StepRecord is a log line
                # and every field has to survive to_json(). storing the dataclass
                # made to_json raise TypeError on the first step that had a
                # checkpoint, which no test caught because none both ran
                # checkpoints and serialised the result.
                checkpoint_results=[
                    result.to_dict()
                    for result in evaluate_checkpoints(env, config.checkpoints)
                ],
                event_labels=[],
                judge_labels=[],
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                latency_s=latency_s,
                intervention_branch=intervention_branch,
                parent_run_id=parent_run_id,
                branch_step=branch_step,
            )
        )

        if is_sentinel or _environment_reports_completion(env):
            break

        messages.append({"role": "assistant", "content": response.text})
        if tool_result is not None:
            messages.append({"role": "user", "content": tool_result})

    return records


def _environment_reports_completion(env: Environment) -> bool:
    """Ask the environment whether the episode is over, if it can answer.

    Optional rather than part of the ``Environment`` protocol: the synthetic
    environments have no notion of task completion, and requiring it would force
    a meaningless implementation on them.
    """
    completed = getattr(env, "task_completed", None)
    return bool(completed()) if callable(completed) else False

"""Command-line entry point for real model-backed harness runs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from ..config import DEFAULT_MODEL_CONFIG
from ..game.orchestrator import TURN_POLICIES
from ..llm.models import ModelConfig
from .batch import run_experiment_spec
from .schedule import persona_profile_ids
from .spec import ExperimentSpec

DEFAULT_NAMES = ["A", "B", "C", "D", "E", "F"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.harness.cli",
        description="Run real LLM agents in the Werewolf harness and write factual artifacts.",
    )
    parser.add_argument("--experiment-id", default="werewolf-run")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Runs per turn policy (total runs = this value times the policy count).",
    )
    parser.add_argument(
        "--role-layout-mode",
        choices=("legacy", "fixed", "counterbalanced"),
        default="legacy",
        help="Hold roles fixed or cross each layout with a complete identity/persona cycle.",
    )
    parser.add_argument(
        "--role-layout-seed",
        type=int,
        help="Independent first role-layout seed (defaults to --seed + 1).",
    )
    parser.add_argument(
        "--role-layout-count",
        type=int,
        help="Number of layouts in a balanced design; inferred when omitted.",
    )
    parser.add_argument(
        "--persona-mode",
        choices=("legacy", "fixed", "randomized", "counterbalanced"),
        default="legacy",
        help="Explicit deterministic persona assignment design.",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        help="Independent persona assignment seed (defaults to a namespaced seed).",
    )
    parser.add_argument(
        "--persona-profiles",
        help="Comma-separated profile IDs; omit to use the full versioned catalog.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Base reproducibility seed.")
    parser.add_argument(
        "--turn-policies",
        default="fixed_round_robin",
        help="Comma-separated policies: fixed_round_robin,bid_reply",
    )
    parser.add_argument("--policy-order", choices=("sequential", "abba"), default="sequential")
    parser.add_argument(
        "--seat-permutation",
        choices=("fixed", "cyclic"),
        default="fixed",
        help="Rotate player identities across seats by paired case while preserving policy pairing.",
    )
    parser.add_argument("--names", default=",".join(DEFAULT_NAMES))
    parser.add_argument("--max-speak-rounds", type=int, default=3)
    parser.add_argument("--max-consecutive-decision-failures", type=int, default=3)
    parser.add_argument("--max-consecutive-no-progress-rounds", type=int, default=3)
    parser.add_argument("--max-game-rounds", type=int, default=20)
    parser.add_argument("--run-timeout", type=float, default=900.0)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--summary-jsonl", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    names = [name.strip() for name in args.names.split(",") if name.strip()]
    if len(names) < 6 or len(names) > 12:
        raise SystemExit("--names must contain 6 to 12 comma-separated names")
    policies = [policy.strip() for policy in args.turn_policies.split(",") if policy.strip()]
    invalid = [policy for policy in policies if policy not in TURN_POLICIES]
    if invalid:
        raise SystemExit(f"unknown turn policies: {invalid}; allowed={list(TURN_POLICIES)}")
    if args.policy_order == "abba" and (len(policies) != 2 or args.runs % 2):
        raise SystemExit("ABBA requires exactly two policies and an even --runs value")
    if args.role_layout_count is not None and args.role_layout_count < 1:
        raise SystemExit("--role-layout-count must be at least 1")
    if args.role_layout_mode == "legacy" and (
        args.role_layout_seed is not None or args.role_layout_count is not None
    ):
        raise SystemExit(
            "--role-layout-seed/count require a non-legacy --role-layout-mode"
        )
    if args.role_layout_mode == "fixed" and args.role_layout_count not in {None, 1}:
        raise SystemExit("fixed --role-layout-mode supports exactly one layout")
    profiles = None
    if args.persona_profiles is not None:
        profiles = [
            profile.strip()
            for profile in args.persona_profiles.split(",")
            if profile.strip()
        ]
        allowed_profiles = set(persona_profile_ids())
        unknown = [profile for profile in profiles if profile not in allowed_profiles]
        if not profiles or len(set(profiles)) != len(profiles) or unknown:
            raise SystemExit(
                "--persona-profiles must contain unique known IDs; "
                f"allowed={list(persona_profile_ids())}"
            )
    if args.persona_mode == "legacy" and (
        args.persona_seed is not None or profiles is not None
    ):
        raise SystemExit(
            "--persona-seed/profiles require a non-legacy --persona-mode"
        )
    seat_cycle = len(names) if args.seat_permutation == "cyclic" else 1
    persona_cycle = (
        len(profiles or persona_profile_ids())
        if args.persona_mode == "counterbalanced"
        else 1
    )
    control_cycle = seat_cycle * persona_cycle
    if args.persona_mode == "counterbalanced" and args.runs % control_cycle:
        raise SystemExit(
            "counterbalanced personas require --runs to be a multiple of the "
            f"complete seat-by-persona control cycle ({control_cycle})"
        )
    if args.role_layout_mode == "counterbalanced":
        if args.runs % control_cycle:
            raise SystemExit(
                "counterbalanced role layouts require --runs to be a multiple "
                f"of the complete control cycle ({control_cycle})"
            )
        if (
            args.role_layout_count is not None
            and (args.runs // control_cycle) % args.role_layout_count
        ):
            raise SystemExit(
                "--runs must contain equal complete cycles for every role layout"
            )
    args.names = names
    args.turn_policies = policies
    args.persona_profiles = profiles
    return args


async def _run(args: argparse.Namespace) -> int:
    model_config = ModelConfig(**DEFAULT_MODEL_CONFIG)
    missing = [
        field for field, value in (
            ("WEREWOLF_LLM_MODEL", model_config.model),
            ("WEREWOLF_LLM_API_KEY", model_config.api_key),
        )
        if not value
    ]
    if missing:
        raise SystemExit("missing real model configuration: " + ", ".join(missing))
    metadata = {"source": "harness_cli"}
    if args.seat_permutation != "fixed":
        metadata["seat_permutation_mode"] = args.seat_permutation
    if args.role_layout_mode != "legacy":
        metadata["role_layout_mode"] = args.role_layout_mode
        if args.role_layout_seed is not None:
            metadata["role_layout_seed"] = args.role_layout_seed
        if args.role_layout_count is not None:
            metadata["role_layout_count"] = args.role_layout_count
    if args.persona_mode != "legacy":
        metadata["persona_mode"] = args.persona_mode
        if args.persona_seed is not None:
            metadata["persona_seed"] = args.persona_seed
        if args.persona_profiles is not None:
            metadata["persona_profile_ids"] = args.persona_profiles
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        player_names=args.names,
        turn_policies=args.turn_policies,
        replicates=args.runs,
        base_seed=args.seed,
        policy_order=args.policy_order,
        max_speak_rounds=args.max_speak_rounds,
        max_consecutive_decision_failures=args.max_consecutive_decision_failures,
        max_consecutive_no_progress_rounds=args.max_consecutive_no_progress_rounds,
        max_game_rounds=args.max_game_rounds,
        run_timeout_seconds=args.run_timeout,
        metadata=metadata,
    )
    batch = await run_experiment_spec(
        spec,
        model_config=model_config,
        artifact_root=args.artifact_root,
        summary_jsonl=args.summary_jsonl,
        resume_jsonl=args.resume,
    )
    print(json.dumps(batch.summary.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return 0 if batch.failed_runs == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

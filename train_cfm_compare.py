"""Train a preprocessing-matched conditional flow-matching comparator."""

from __future__ import annotations

import argparse

from src.rcfm.training import run_training
from train_rcfm import (
    build_argparser as build_shared_argparser,
    build_datasets,
    parse_args_with_config as parse_shared_args_with_config,
)


MODEL_FAMILY = "CFM"


def build_argparser() -> argparse.ArgumentParser:
    parser = build_shared_argparser()
    parser.description = __doc__
    parser.set_defaults(
        region_weight=0.0,
        use_minibatch_ot=False,
        wandb_group="cfm-compare",
    )
    return parser


def validate_compare_contract(args: argparse.Namespace) -> None:
    if args.flow_matcher != "conditional" or float(args.sigma) != 0.0:
        raise ValueError("CFM compare requires the canonical conditional path with sigma=0")
    if float(args.region_weight) != 0.0:
        raise ValueError("CFM compare requires region_weight=0")
    if bool(args.use_minibatch_ot):
        raise ValueError("CFM compare requires minibatch OT to be disabled")


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    args = parse_shared_args_with_config(argv, parser=build_argparser())
    validate_compare_contract(args)
    args.model_family = MODEL_FAMILY
    return args


def train(args: argparse.Namespace) -> None:
    validate_compare_contract(args)
    args.model_family = MODEL_FAMILY
    run_training(args, build_datasets)


if __name__ == "__main__":
    parsed_args = parse_args_with_config()
    if not parsed_args.data_root:
        raise ValueError("--data_root or RCFM_DATA_ROOT is required")
    if not parsed_args.output_dir:
        raise ValueError("--output_dir or RCFM_RUNS_ROOT is required")
    train(parsed_args)

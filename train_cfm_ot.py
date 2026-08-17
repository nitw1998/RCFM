"""Train the canonical CFM model with exact-assignment minibatch OT coupling."""

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
        use_minibatch_ot=True,
        ot_method="exact",
        ot_strict_mode=True,
        ot_sampling_strategy="assignment",
        wandb_group="cfm-ot",
    )
    return parser


def validate_cfm_ot_contract(args: argparse.Namespace) -> None:
    if args.flow_matcher != "conditional" or float(args.sigma) != 0.0:
        raise ValueError("CFM-OT requires the canonical conditional path with sigma=0")
    if float(args.region_weight) != 0.0:
        raise ValueError("CFM-OT requires region_weight=0")
    if not bool(args.use_minibatch_ot):
        raise ValueError("CFM-OT requires minibatch OT enabled")
    if args.ot_method != "exact" or args.ot_sampling_strategy != "assignment":
        raise ValueError("CFM-OT requires exact OT with assignment sampling")
    if not bool(args.ot_strict_mode):
        raise ValueError("CFM-OT requires strict OT validation")


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    args = parse_shared_args_with_config(argv, parser=build_argparser())
    validate_cfm_ot_contract(args)
    args.model_family = MODEL_FAMILY
    return args


def train(args: argparse.Namespace) -> None:
    validate_cfm_ot_contract(args)
    args.model_family = MODEL_FAMILY
    run_training(args, build_datasets)


if __name__ == "__main__":
    parsed_args = parse_args_with_config()
    if not parsed_args.data_root:
        raise ValueError("--data_root or RCFM_DATA_ROOT is required")
    if not parsed_args.output_dir:
        raise ValueError("--output_dir or RCFM_RUNS_ROOT is required")
    train(parsed_args)

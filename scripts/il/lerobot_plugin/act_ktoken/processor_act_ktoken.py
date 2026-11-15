"""Pre/post-processing of act_ktoken: the same as ACT's (normalisation of the whole state, K included,
happens before the model splits K off)."""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_act_ktoken import ACTKTokenConfig


def make_act_ktoken_pre_post_processors(
    config: ACTKTokenConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
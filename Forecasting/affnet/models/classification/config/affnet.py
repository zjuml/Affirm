# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

from typing import Dict

from options import logger
# from venv import logger


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.affnet.mode", "small")
    if mode is None:
        logger.error("Please specify mode")

    head_dim = getattr(opts, "model.classification.affnet.head_dim", None)
    num_heads = getattr(opts, "model.classification.affnet.number_heads", 4)
    if head_dim is not None:
        if num_heads is not None:
            logger.error(
                "--model.classification.affnet.head-dim and --model.classification.affnet.number-heads "
                "are mutually exclusive."
            )
    elif num_heads is not None:
        if head_dim is not None:
            logger.error(
                "--model.classification.affnet.head-dim and --model.classification.affnet.number-heads "
                "are mutually exclusive."
            )
    mode = mode.lower()
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer4": {  # 14x14
                "out_channels": 104,
                "transformer_channels": 104,
                "ffn_dim": 208,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer5": {  # 7x7
                "out_channels": 144,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer4": {  # 14x14
                "out_channels": 160,
                "transformer_channels": 160,
                "ffn_dim": 320,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer5": {  # 7x7
                "out_channels": 192,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 128,
                "transformer_channels": 128,
                "ffn_dim": 256,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer4": {  # 14x14
                "out_channels": 256,
                "transformer_channels": 256,
                "ffn_dim": 512,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer5": {  # 7x7
                "out_channels": 320,
                "transformer_channels": 320,
                "ffn_dim": 640,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "base":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 128,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 256,
                "transformer_channels": 256,
                "ffn_dim": 512,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer4": {  # 14x14
                "out_channels": 512,
                "transformer_channels": 512,
                "ffn_dim": 1024,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
                "no_fuse": True
            },
            "layer5": {  # 7x7
                "out_channels": 640,
                "transformer_channels": 640,
                "ffn_dim": 1280,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
                "no_fuse": True
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "large":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 2,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 128,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 6,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 256,
                "transformer_channels": 256,
                "ffn_dim": 512,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
            },
            "layer4": {  # 14x14
                "out_channels": 512,
                "transformer_channels": 512,
                "ffn_dim": 1024,
                "transformer_blocks": 18,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
                "no_fuse": True
            },
            "layer5": {  # 7x7
                "out_channels": 768,
                "transformer_channels": 768,
                "ffn_dim": 1536,
                "transformer_blocks": 6,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "aff_block",
                "no_fuse": True
            },
            "last_layer_exp_factor": 4,
        }
    else:
        raise NotImplementedError

    return config
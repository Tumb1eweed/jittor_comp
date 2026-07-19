import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jittor import nn

from models.pgd import PGDModel
from tools.train_shapenet_one_epoch import freeze_batchnorm_stats, select_trainable_parameters


def test_freeze_batchnorm_stats_sets_only_batchnorm_modules_to_eval():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.BatchNorm1d(4),
        nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)),
    )
    model.train()

    frozen = freeze_batchnorm_stats(model)

    assert frozen == 2
    assert model[0].is_training()
    assert not model[1].is_training()
    assert model[2][0].is_training()
    assert not model[2][1].is_training()


class HeadOnlyArgs:
    pgd_train_head_only = True


class DecoderHeadArgs(HeadOnlyArgs):
    pgd_train_head_only = False
    pgd_train_decoder_head_only = True


def test_select_trainable_parameters_can_limit_to_pgd_output_head():
    model = PGDModel(args=HeadOnlyArgs())

    params, names = select_trainable_parameters(model, HeadOnlyArgs())

    assert len(params) == 5
    assert names == [
        "feature_nets.linear0_1.weight",
        "feature_nets.linear0_2.weight",
        "feature_nets.linear0_2.bias",
        "feature_nets.linear0_3.weight",
        "feature_nets.linear0_3.bias",
    ]


def test_select_trainable_parameters_can_limit_to_pgd_decoder_codebook_and_head():
    model = PGDModel(args=DecoderHeadArgs())

    params, names = select_trainable_parameters(model, DecoderHeadArgs())

    assert len(params) > 5
    assert any(name.startswith("feature_nets.decoder_blocks.") for name in names)
    assert any(name.startswith("feature_nets.codebooks.") for name in names)
    assert "feature_nets.linear0_3.bias" in names
    assert not any(name.startswith("feature_nets.encoder_blocks.") for name in names)
    assert not any(name.endswith(".running_mean") or name.endswith(".running_var") for name in names)


if __name__ == "__main__":
    test_freeze_batchnorm_stats_sets_only_batchnorm_modules_to_eval()
    test_select_trainable_parameters_can_limit_to_pgd_output_head()
    test_select_trainable_parameters_can_limit_to_pgd_decoder_codebook_and_head()
    os._exit(0)

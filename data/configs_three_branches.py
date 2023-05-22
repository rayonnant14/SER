from trainer import TrainerThreeBranches
from models import ThreeBranchesClassification
from models import (
    ConcatinationBasedFusion,
    WightedSumBasedFusion,
    WeighedFusionV1,
    WeighedFusionV2,
    MulFusion,
    WeighedMulFusion,
    AttentionBasedFusion,
    LateFusionV1,
    LateFusionV2,
)

fusion_firsts = [
    ConcatinationBasedFusion,
    WightedSumBasedFusion,
    WeighedFusionV1,
    WeighedFusionV2,
    MulFusion,
    WeighedMulFusion,
    AttentionBasedFusion,
]
fusion_seconds = fusion_firsts + [LateFusionV1, LateFusionV2]
base_config_three_branches = {
    "trainer": TrainerThreeBranches,
    "use_keys": ["x", "x_asr", "x_lm", "y"],
}

configs_three_branches = []
for fusion_first in fusion_firsts:
    for fusion_second in fusion_seconds:
        trainer_inputs = {
            "trainer_inputs": {
                "features_num": 512,
                "features_num_second": 256,
                "model_class": ThreeBranchesClassification,
                "fusion_first": fusion_first,
                "fusion_second": fusion_second,
            }
        }
        input_parameters = {
            **base_config_three_branches,
            **trainer_inputs,
        }
        configs_three_branches.append(input_parameters)
        # print(configs_three_branches)

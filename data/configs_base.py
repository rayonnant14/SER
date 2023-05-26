from trainer import TrainerClassification, TrainerOneBranch, TrainerTwoBranches
from models import (
    TIMNETClassification,
    AdditionalFeatureClassification,
    TwoBranchesClassification,
)
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

configs_timnet = [
    {
        "trainer": TrainerOneBranch,
        "use_keys": ["x_asr", "y"],
        "trainer_inputs": {
             "features_num": 1024,
              "model_class": AdditionalFeatureClassification,
        }
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": ConcatinationBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
]
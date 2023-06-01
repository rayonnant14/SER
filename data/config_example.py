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

configs = [
    {
        "trainer": TrainerClassification,
        "use_keys": ["x", "y"],
        "trainer_inputs": {
            "model_class": TIMNETClassification,
        },
    }
]

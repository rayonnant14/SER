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
    },
    {
        "trainer": TrainerOneBranch,
        "use_keys": ["x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "model_class": AdditionalFeatureClassification,
        },
    },
    {
        "trainer": TrainerOneBranch,
        "use_keys": ["x_asr", "y"],
        "trainer_inputs": {
             "features_num": 1024,
              "model_class": AdditionalFeatureClassification,
        }
    },
    {
        "trainer": TrainerOneBranch,
        "use_keys": ["x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "model_class": AdditionalFeatureClassification,
        },
    },
        {
        "trainer": TrainerOneBranch,
        "use_keys": ["x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "model_class": AdditionalFeatureClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": ConcatinationBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": WightedSumBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": WeighedFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": WeighedFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": MulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": WeighedMulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": AttentionBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": LateFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_opensmile", "y"],
        "trainer_inputs": {
            "features_num": 988,
            "fusion": LateFusionV2,
            "model_class": TwoBranchesClassification,
        },
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
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": WightedSumBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": WeighedFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": WeighedFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": MulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": WeighedMulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": AttentionBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": LateFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_asr", "y"],
        "trainer_inputs": {
            "features_num": 1024,
            "fusion": LateFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": ConcatinationBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WightedSumBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": MulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedMulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": AttentionBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": LateFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_xlm_roberta", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": LateFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
        {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": ConcatinationBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WightedSumBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": MulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": WeighedMulFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": AttentionBasedFusion,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": LateFusionV1,
            "model_class": TwoBranchesClassification,
        },
    },
    {
        "trainer": TrainerTwoBranches,
        "use_keys": ["x", "x_bert", "y"],
        "trainer_inputs": {
            "features_num": 768,
            "fusion": LateFusionV2,
            "model_class": TwoBranchesClassification,
        },
    },
]

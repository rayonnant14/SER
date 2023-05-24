from trainer import TrainerClassification
from models import (
    TIMNETClassification,
)

configs_timnet = [
    {
        "trainer": TrainerClassification,
        "use_keys": ["x", "y"],
        "trainer_inputs": {
            "model_class": TIMNETClassification,
        },
    }
]
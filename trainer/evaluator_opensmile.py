import torch
from trainer import EvaluatorClassification


class EvaluatorOpenSmile(EvaluatorClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_weights(self, model, fold):
        model_path = self.save_path + "opensmile_" + str(fold) + ".pth"
        model.load_state_dict(torch.load(model_path))

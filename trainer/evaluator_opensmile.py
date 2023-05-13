import torch
from trainer import EvaluatorClassification


class EvaluatorOpenSmile(EvaluatorClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_weights(self, model, fold):
        model_path = self.save_path + "timnet_opensmile" + str(fold) + ".pth"
        model.load_state_dict(torch.load(model_path))

    def infer_model(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        return labels, out

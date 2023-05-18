from trainer import EvaluatorOpenSmile


class EvaluatorTwoBranches(EvaluatorOpenSmile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validation_step(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        return labels, out

from trainer import EvaluatorOpenSmile


class EvaluatorASR_LM(EvaluatorOpenSmile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validation_step(self, model, batch):
        x, x_asr, x_lm, labels = batch
        x, x_asr, x_lm, labels = (
            x.to(self.device),
            x_asr.to(self.device),
            x_lm.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_asr, x_lm)
        return labels, out

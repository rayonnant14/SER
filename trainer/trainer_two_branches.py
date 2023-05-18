from trainer import TrainerOpenSmile


class TrainerTwoBranches(TrainerOpenSmile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

    def validation_step(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        return labels, out

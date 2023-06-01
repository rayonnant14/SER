from trainer import TrainerOneBranch


class TrainerThreeBranches(TrainerOneBranch):
    def __init__(
        self, features_num_second, fusion_first, fusion_second, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.features_num_second = features_num_second
        self.fusion_first = fusion_first
        self.fusion_second = fusion_second

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            dilations=self.dilations,
            features_num_first=self.features_num,
            features_num_second=self.features_num_second,
            fusion_first=self.fusion_first,
            fusion_second=self.fusion_second,
        )
        return model

    def training_step(self, model, batch):
        x, x_asr, x_lm, labels = batch
        x, x_asr, x_lm, labels = (
            x.to(self.device),
            x_asr.to(self.device),
            x_lm.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_asr, x_lm)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

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

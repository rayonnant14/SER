from trainer import TrainerOneBranch


class TrainerTwoBranches(TrainerOneBranch):
    def __init__(self, fusion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion = fusion

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            features_num=self.features_num,
            fusion=self.fusion,
            with_pca=self.with_pca,
            pca_components=self.pca_components,
        )
        return model
    
    def training_step(self, model, batch):
        x, additional_feature, labels = batch
        x, additional_feature, labels = (
            x.to(self.device),
            additional_feature.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, additional_feature)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

    def validation_step(self, model, batch):
        x, additional_feature, labels = batch
        x, additional_feature, labels = (
            x.to(self.device),
            additional_feature.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, additional_feature)
        return labels, out

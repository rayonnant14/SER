from trainer import TrainerClassification


class TrainerOpenSmile(TrainerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            with_pca=self.with_pca,
            pca_components=self.pca_components
        )
        return model

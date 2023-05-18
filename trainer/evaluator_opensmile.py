from trainer import EvaluatorClassification


class EvaluatorOpenSmile(EvaluatorClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            with_pca=self.with_pca,
        )
        return model

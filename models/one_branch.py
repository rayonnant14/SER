import torch.nn as nn


class AdditionalFeature(nn.Module):
    def __init__(
        self,
        features_num,
        dropout_rate=0.1,
        with_pca=False,
        pca_components=100
    ):
        super().__init__()
        if with_pca:
            self.features_num = pca_components
        else:
            self.features_num = features_num

        # self.layers = nn.Sequential(
        #     nn.Conv1d(
        #         1,
        #         1,
        #         kernel_size=1,
        #     ),
        #     nn.ReLU(),
        #     nn.Dropout1d(dropout_rate),
        #     nn.BatchNorm1d(num_features=1),
        #     nn.Conv1d(
        #         1,
        #         1,
        #         kernel_size=1,
        #     ),
        #     nn.ReLU(),
        #     nn.Dropout1d(dropout_rate),
        #     nn.BatchNorm1d(num_features=1),
        # )
        self.layers = nn.Sequential(
            nn.Linear(self.features_num, self.features_num),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.BatchNorm1d(num_features=self.features_num),
            nn.Linear(
                self.features_num, self.features_num // 2
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.BatchNorm1d(num_features=self.features_num // 2),
        )

    def forward(self, additional_x):
        output = self.layers(additional_x)
        return output


class AdditionalFeatureClassification(nn.Module):
    def __init__(
        self,
        class_num,
        features_num,
        dropout_rate=0.1,
        with_pca=False,
        pca_components=100,
        model_name='additional_feature_classification',
    ):
        super().__init__()
        self.model_name = model_name
        if with_pca:
            self.features_num = pca_components
        else:
            self.features_num = features_num

        self.AdditionalFeature = AdditionalFeature(
            features_num=self.features_num,
            dropout_rate=dropout_rate,
            with_pca=with_pca,
            pca_components=pca_components
        )

        self.classification = nn.Sequential(
            nn.Linear(
                self.features_num // 2,
                self.features_num // 4,
            ),
            nn.ReLU(),
            nn.Linear(self.features_num // 4, class_num),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x_additional):
        output = self.AdditionalFeature(x_additional)
        output = self.classification(output)
        # x = self.softmax(x)
        return output

    def get_name(self):
        return self.model_name

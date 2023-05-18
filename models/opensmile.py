import torch.nn as nn


class OpenSmile(nn.Module):
    def __init__(
        self,
        class_num,
        dropout_rate=0.1,
        opensmile_features_num=988,
        with_pca=False,
        pca_components=100
    ):
        super().__init__()
        if with_pca:
            self.opensmile_features_num = pca_components
        else:
            self.opensmile_features_num = opensmile_features_num
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
            nn.Linear(self.opensmile_features_num, self.opensmile_features_num),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.BatchNorm1d(num_features=self.opensmile_features_num),
            nn.Linear(
                self.opensmile_features_num, self.opensmile_features_num // 2
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.BatchNorm1d(num_features=self.opensmile_features_num // 2),
        )

    def forward(self, opensmile_x):
        output = self.layers(opensmile_x)
        return output


class OpenSmileClassification(nn.Module):
    def __init__(
        self,
        class_num,
        dropout_rate=0.1,
        opensmile_features_num=988,
        with_pca=False,
        pca_components=100
    ):
        super().__init__()
        if with_pca:
            self.opensmile_features_num = pca_components
        else:
            self.opensmile_features_num = opensmile_features_num

        self.OpenSmile = OpenSmile(
            class_num=class_num,
            dropout_rate=dropout_rate,
            opensmile_features_num=self.opensmile_features_num,
            with_pca=with_pca,
            pca_components=pca_components
        )

        self.classification = nn.Sequential(
            nn.Linear(
                self.opensmile_features_num // 2,
                self.opensmile_features_num // 4,
            ),
            nn.ReLU(),
            nn.Linear(self.opensmile_features_num // 4, class_num),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, opensmile_x):
        output = self.OpenSmile(opensmile_x)
        output = self.classification(output)
        # x = self.softmax(x)
        return output

    def get_name(self):
        return "opensmile"

import torch.nn as nn

class OpenSmileClassification(nn.Module):
    def __init__(
        self,
        class_num,
        dropout_rate = 0.1,
        opensmile_features_num=988,
    ):
        super().__init__()
        self.opensmile_features_num = opensmile_features_num
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

        self.classification = nn.Sequential(
            nn.Linear(self.opensmile_features_num // 2, self.opensmile_features_num // 4),
            nn.ReLU(),
            nn.Linear(self.opensmile_features_num // 4, class_num),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, opensmile_x):
        output = self.layers(opensmile_x)
        output = self.classification(output)
        # x = self.softmax(x)
        return output

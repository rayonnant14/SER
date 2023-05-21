import torch.nn as nn

from models import TIMNET, AdditionalFeature


class TwoBranchesClassification(nn.Module):
    def __init__(
        self,
        class_num,
        features_num,
        fusion,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
        with_pca=False,
        pca_components=100
    ):
        super().__init__()
        if with_pca:
            self.features_num = pca_components
        else:
            self.features_num = features_num

        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        # First Branch
        self.first_branch = TIMNET(
            class_num=class_num,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            dropout_rate=dropout_rate,
        )

        # Second Branch
        self.second_branch = AdditionalFeature(
            dropout_rate=dropout_rate,
            features_num=self.features_num,
            with_pca=with_pca,
            pca_components=pca_components
        )

        self.fusion = fusion(
            embedding_first_size=nb_filters,
            embedding_second_size=self.features_num // 2,
            class_num=class_num,
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_additional):
        output_first_branch = self.first_branch(x)
        output_second_branch = self.second_branch(x_additional)

        output = self.fusion(output_first_branch, output_second_branch)
        # x = self.softmax(x)
        return output

    def get_name(self):
        return "two_branches"

import torch.nn as nn

from models import TIMNET, OpenSmile

from models import (
    ConcatinationBasedFusion,
    WightedSumBasedFusion,
    WeighedFusionV1,
    WeighedFusionV2,
    MulFusion,
    AttentionBasedFusion,
    LateFusionV1,
    LateFusionV2,
)


class TwoBranches(nn.Module):
    def __init__(
        self,
        class_num,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
        opensmile_features_num=988,
        with_pca=False,
    ):
        super().__init__()
        if with_pca:
            self.opensmile_features_num = 100
        else:
            self.opensmile_features_num = opensmile_features_num

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
        self.second_branch = OpenSmile(
            class_num=class_num,
            dropout_rate=dropout_rate,
            opensmile_features_num=opensmile_features_num,
            with_pca=with_pca,
        )

        self.fusion = LateFusionV2(
            embedding_first_size=nb_filters,
            embedding_second_size=self.opensmile_features_num // 2,
            class_num=class_num,
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, opensmile_x):
        output_first_branch = self.first_branch(x)
        output_second_branch = self.second_branch(opensmile_x)

        output = self.fusion(output_first_branch, output_second_branch)
        # x = self.softmax(x)
        return output

    def get_name(self):
        return "two_branches"

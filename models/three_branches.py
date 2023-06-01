# import torch.nn as nn
# import torch

# from models import TIMNET
# from models import (
#     ConcatinationBasedFusion,
#     WightedSumBasedFusion,
#     WeighedFusionV1,
#     WeighedFusionV2,
#     MulFusion,
#     WeighedMulFusion,
#     AttentionBasedFusion,
#     LateFusionV1,
#     LateFusionV2,
# )


# class ASR_LM(nn.Module):
#     def __init__(
#         self,
#         class_num,
#         nb_filters=39,
#         kernel_size=2,
#         nb_stacks=1,
#         dilations=8,
#         dropout_rate=0.1,
#         asr_features_num=512,
#         lm_features_num=256,
#         with_pca=False,
#         pca_components=100,
#     ):
#         super().__init__()
#         self.dropout_rate = dropout_rate
#         self.dilations = dilations
#         self.nb_stacks = nb_stacks
#         self.kernel_size = kernel_size
#         self.nb_filters = nb_filters

#         # First Branch
#         self.first_branch = TIMNET(
#             class_num=class_num,
#             nb_filters=nb_filters,
#             kernel_size=kernel_size,
#             nb_stacks=nb_stacks,
#             dilations=dilations,
#             dropout_rate=dropout_rate,
#         )
#         # Second Branch
#         # self.second_branch = nn.Conv1d(
#         #     in_channels=2, out_channels=1, kernel_size=1
#         # )
#         self.fusion = WeighedFusionV2(
#             embedding_first_size=nb_filters,
#             embedding_second_size=asr_features_num + lm_features_num,
#             class_num=class_num,
#         )
#         # self.softmax = nn.Softmax(dim=1)

#     def forward(self, x, x_asr, x_lm):
#         output_first_branch = self.first_branch(x)
#         x_second_branch = torch.cat([x_asr, x_lm], dim=1)
#         # output_second_branch = output_second_branch.reshape(
#         #     output_second_branch.size(0), -1
#         # )
#         output = self.fusion(output_first_branch, x_second_branch)
#         # x = self.softmax(x)
#         return output

#     def get_name(self):
#         return "asr_lm_timnet"

import torch.nn as nn

from models import TIMNET, AdditionalFeature


class ThreeBranchesClassification(nn.Module):
    def __init__(
        self,
        class_num,
        features_num_first,
        features_num_second,
        fusion_first,
        fusion_second,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
        # with_pca=False,
        # pca_components=100
    ):
        super().__init__()
        # if with_pca:
        #     self.features_num = pca_components
        # else:
        #     self.features_num = features_num
        self.features_num_first = features_num_first
        self.features_num_second = features_num_second
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
            features_num=self.features_num_first,
            with_pca=False,
            pca_components=None,
        )

        # Third Branch
        self.third_branch = AdditionalFeature(
            dropout_rate=dropout_rate,
            features_num=self.features_num_second,
            with_pca=False,
            pca_components=None,
        )

        self.fusion_first = fusion_first(
            embedding_first_size=self.features_num_first // 2,
            embedding_second_size=self.features_num_second // 2,
            class_num=class_num,
            return_embedding=True,
        )

        self.fusion_second = fusion_second(
            embedding_first_size=nb_filters,
            embedding_second_size=self.fusion_first.total_embedding_size,
            class_num=class_num,
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_additional_first, x_additional_second):
        output_first_branch = self.first_branch(x)
        output_second_branch = self.second_branch(x_additional_first)
        output_third_branch = self.third_branch(x_additional_second)

        output_first = self.fusion_first(output_second_branch, output_third_branch)
        output_second = self.fusion_second(output_first_branch, output_first)
        # x = self.softmax(x)
        return output_second

    def get_name(self):
        return "three_branches"

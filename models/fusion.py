import torch
import torch.nn as nn


class ConcatinationBasedFusion(nn.Module):
    def __init__(self, embedding_first_size, embedding_second_size, class_num):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        total_embedding_size = embedding_first_size + embedding_second_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        output = torch.cat([embedding_first, embedding_second], dim=1)
        output = self.classification(output)
        return output


class WightedSumBasedFusion(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
        embedding_first_weight=0.5,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.embedding_first_weight = embedding_first_weight
        self.embegging_second_weight = 1.0 - embedding_first_weight
        self.dim_conversion = nn.Linear(
            self.embedding_first_size, self.embedding_second_size
        )

        total_embedding_size = self.embedding_second_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_first = self.dim_conversion(embedding_first)
        sum = torch.add(
            self.embedding_first_weight * embedding_first,
            self.embegging_second_weight * embedding_second,
        )

        output = self.classification(sum)
        return output


class WeighedFusionV1(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.dim_conversion = nn.Linear(
            self.embedding_first_size,
            self.embedding_second_size,
        )
        self.aggregation_layer = torch.nn.Conv1d(
            in_channels=2, out_channels=1, kernel_size=1
        )
        total_embedding_size = self.embedding_second_size

        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_first = self.dim_conversion(embedding_first)

        embedding_first = embedding_first.reshape(
            embedding_first.size(0), 1, -1
        )
        embedding_second = embedding_second.reshape(
            embedding_second.size(0), 1, -1
        )

        concat_embeddings = torch.cat(
            [embedding_first, embedding_second], dim=1
        )
        weighed_embeggings = self.aggregation_layer(concat_embeddings)
        weighed_embeggings = weighed_embeggings.reshape(
            weighed_embeggings.size(0), -1
        )
        output = self.classification(weighed_embeggings)
        return output


class WeighedFusionV2(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.dim_conversion = nn.Linear(
            self.embedding_first_size,
            self.embedding_second_size,
        )
        self.weight_first = nn.Parameter(
            torch.randn(self.embedding_second_size)
        )
        self.weight_second = nn.Parameter(
            torch.randn(self.embedding_second_size)
        )

        total_embedding_size = self.embedding_second_size

        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_first = self.dim_conversion(embedding_first)

        weight_first_sigmoid = self.weight_first.sigmoid()
        weight_second_sigmoid = self.weight_second.sigmoid()

        sum = torch.add(
            embedding_first * weight_first_sigmoid,
            embedding_second * weight_second_sigmoid,
        )
        output = self.classification(sum)
        return output


class MulFusion(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.dim_conversion = nn.Linear(
            self.embedding_first_size,
            self.embedding_second_size,
        )
        total_embedding_size = self.embedding_second_size

        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_first = self.dim_conversion(embedding_first)

        mul = torch.mul(embedding_first, embedding_second)
        output = self.classification(mul)
        return output


class WeighedMulFusion(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.weight_matrix = nn.Parameter(
            torch.randn(embedding_second_size, embedding_first_size)
        )

        total_embedding_size = embedding_first_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_second = torch.matmul(embedding_second, self.weight_matrix)

        mul = torch.mul(embedding_first, embedding_second)
        output = self.classification(mul)
        return output


class AttentionBasedFusion(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.embedding_first_size = embedding_first_size
        self.embedding_second_size = embedding_second_size
        self.dim_conversion = nn.Linear(
            self.embedding_first_size,
            self.embedding_second_size,
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_second_size, num_heads=13
        )
        total_embedding_size = self.embedding_second_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_first = self.dim_conversion(embedding_first)

        embedding_first = embedding_first.reshape(
            embedding_first.size(0), 1, -1
        )
        embedding_second = embedding_second.reshape(
            embedding_second.size(0), 1, -1
        )

        attn_output, _ = self.multihead_attn(
            query=embedding_first,
            key=embedding_second,
            value=embedding_first,
            average_attn_weights=True,
            need_weights=False,
        )
        attn_output = attn_output.reshape(attn_output.size(0), -1)
        output = self.classification(attn_output)
        return output


class LateFusionV1(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.classification_first_branch = nn.Sequential(
            nn.Linear(embedding_first_size, class_num), nn.Softmax(dim=1)
        )
        self.classification_second_branch = nn.Sequential(
            nn.Linear(embedding_second_size, embedding_second_size // 4),
            nn.ReLU(),
            nn.Linear(embedding_second_size // 4, class_num),
            nn.Softmax(dim=1),
        )
        self.weight_first = nn.Parameter(torch.randn(class_num))
        self.weight_second = nn.Parameter(torch.randn(class_num))

    def forward(self, embedding_first, embedding_second):
        output_first_branch = self.classification_first_branch(embedding_first)
        output_second_branch = self.classification_second_branch(
            embedding_second
        )

        weight_first_sigmoid = self.weight_first.sigmoid()
        weight_second_sigmoid = self.weight_second.sigmoid()

        sum = torch.add(
            output_first_branch * weight_first_sigmoid,
            output_second_branch * weight_second_sigmoid,
        )
        return sum


class LateFusionV2(nn.Module):
    def __init__(
        self,
        embedding_first_size,
        embedding_second_size,
        class_num,
    ):
        super().__init__()
        self.classification_first_branch = nn.Sequential(
            nn.Linear(embedding_first_size, class_num), nn.Softmax(dim=1)
        )
        self.classification_second_branch = nn.Sequential(
            nn.Linear(embedding_second_size, embedding_second_size // 4),
            nn.ReLU(),
            nn.Linear(embedding_second_size // 4, class_num),
            nn.Softmax(dim=1),
        )
        self.weight_first = 0.7
        self.weight_second = 0.3

    def forward(self, embedding_first, embedding_second):
        output_first_branch = self.classification_first_branch(embedding_first)
        output_second_branch = self.classification_second_branch(
            embedding_second
        )

        sum = torch.add(
            output_first_branch * self.weight_first,
            output_second_branch * self.weight_second,
        )
        return sum

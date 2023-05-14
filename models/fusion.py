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
            self.embedding_second_size, self.embedding_first_size
        )

        total_embedding_size = self.embedding_first_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        embedding_second = self.dim_conversion(embedding_second)
        sum = torch.add(
            self.embedding_first_weight * embedding_first,
            self.embegging_second_weight * embedding_second,
        )

        output = self.classification(sum)
        return output


class WeighedFusion(nn.Module):
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
            self.embedding_second_size, self.embedding_first_size
        )

        total_embedding_size = self.embedding_first_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        return


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

        total_embedding_size = self.embedding_first_size
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )

    def forward(self, embedding_first, embedding_second):
        return


class LateFusion(nn.Module):
    def __init__(
        self,
        logits_first_size,
        logits_second_size,
    ):
        super().__init__()

    def forward(self, logits_first, logits_second):
        return

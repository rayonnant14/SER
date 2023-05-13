import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Temporal_Aware_Block


class TIMNET_OpenSmile(nn.Module):
    def __init__(
        self,
        class_num,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
        opensmile_features_num=988,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.opensmile_features_num = opensmile_features_num

        # First Branch
        self.forward_convd = nn.Conv1d(
            in_channels=nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
            padding=0,
        )
        self.backward_convd = nn.Conv1d(
            in_channels=nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
            padding=0,
        )
        self.skip_out_forwards = nn.Sequential()
        self.skip_out_backwards = nn.Sequential()
        for s in range(self.nb_stacks):
            for i in [2**i for i in range(self.dilations)]:
                self.skip_out_forwards.append(
                    Temporal_Aware_Block(
                        s,
                        i,
                        self.nb_filters,
                        self.kernel_size,
                        self.dropout_rate,
                    )
                )
                self.skip_out_backwards.append(
                    Temporal_Aware_Block(
                        s,
                        i,
                        self.nb_filters,
                        self.kernel_size,
                        self.dropout_rate,
                    )
                )
        self.pooling = nn.Sequential()
        for i in range(self.dilations):
            self.pooling.append(nn.AdaptiveAvgPool1d(1))
        self.flatten_1 = nn.Flatten(start_dim=-2, end_dim=-1)

        self.weight_layer = nn.Conv1d(
            in_channels=self.dilations, out_channels=1, kernel_size=1
        )
        self.flatten_2 = nn.Flatten(start_dim=-2, end_dim=-1)

        # Second Branch
        self.second_branch = nn.Sequential(
            nn.Linear(self.opensmile_features_num, self.opensmile_features_num),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.opensmile_features_num),
            nn.Linear(
                self.opensmile_features_num, self.opensmile_features_num // 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.opensmile_features_num // 2),
        )

        total_embedding_size = nb_filters + opensmile_features_num // 2
        self.classification = nn.Sequential(
            nn.Linear(total_embedding_size, total_embedding_size // 4),
            nn.ReLU(),
            nn.Linear(total_embedding_size // 4, class_num),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, opensmile_x):
        # First Branch
        forward = x
        backward = torch.flip(x, [1])

        forward_convd = self.forward_convd(forward)
        backward_convd = self.backward_convd(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        for idx, i in enumerate([2**i for i in range(self.dilations)]):
            skip_out_forward = self.skip_out_forwards[idx](skip_out_forward)
            skip_out_backward = self.skip_out_backwards[idx](skip_out_backward)

            temp_skip = skip_out_forward + skip_out_backward
            temp_skip = self.pooling[idx](temp_skip)
            temp_skip = self.flatten_1(temp_skip)
            temp_skip = torch.unsqueeze(temp_skip, 1)
            final_skip_connection.append(temp_skip)

        output_first_branch = torch.cat(final_skip_connection, dim=-2)
        output_first_branch = self.weight_layer(output_first_branch)
        output_first_branch = self.flatten_2(output_first_branch)

        # Second Branch
        output_second_branch = self.second_branch(opensmile_x)

        output = torch.cat([output_first_branch, output_second_branch], dim=1)
        output = self.classification(output)
        # x = self.softmax(x)
        return output

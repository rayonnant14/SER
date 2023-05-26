import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from typing import Union


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        output = self.conv1d.forward(input)
        shift = -self.conv1d.padding[0] if self.conv1d.padding[0] != 0 else None
        output = output[:, :, :shift]
        return output


class SpatialDropout(torch.nn.Module):
    """Spatial dropout module.

    Apply dropout to full channels on tensors of input (B, C, D)
    """

    def __init__(
        self,
        dropout_probability: float = 0.15,
        shape: Optional[Union[tuple, list]] = None,
    ):
        super().__init__()
        if shape is None:
            shape = (0, 2, 1)
        self.dropout = nn.Dropout1d(dropout_probability)
        self.shape = (shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of spatial dropout module."""
        y = x.permute(*self.shape)
        y = self.dropout(y)
        return y.permute(*self.shape)


class Temporal_Aware_Block(nn.Module):
    def __init__(self, s, i, nb_filters, kernel_size, dropout_rate=0):
        super().__init__()
        self.s = s
        self.i = i
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.block_1 = nn.Sequential(
            CausalConv1d(
                in_channels=self.nb_filters,
                out_channels=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation=self.i,
            ),
            nn.BatchNorm1d(num_features=self.nb_filters),
            nn.ReLU(),
            SpatialDropout(dropout_probability=self.dropout_rate),
        )
        self.block_2 = nn.Sequential(
            CausalConv1d(
                in_channels=self.nb_filters,
                out_channels=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation=self.i,
            ),
            nn.BatchNorm1d(num_features=self.nb_filters),
            nn.ReLU(),
            SpatialDropout(dropout_probability=self.dropout_rate),
        )

    def forward(self, x):
        original_x = x
        output_1 = self.block_1(x)
        output_2 = self.block_2(output_1)
        output = torch.sigmoid(output_2)
        F_x = torch.mul(original_x, output)
        return F_x


class TIMNET(nn.Module):
    def __init__(
        self,
        class_num,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
    ):
        super(TIMNET, self).__init__()
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.forward_convd = CausalConv1d(
            in_channels=nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
            # padding=0,
        )
        self.backward_convd = CausalConv1d(
            in_channels=nb_filters,
            out_channels=self.nb_filters,
            kernel_size=1,
            dilation=1,
            # padding=0,
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

        # self.weight_layer = nn.Conv1d(
        #     in_channels=self.dilations, out_channels=1, kernel_size=1
        # )
        self.weight_layer = nn.Parameter(
            torch.randn(self.dilations, 1)
        )
        # self.flatten_2 = nn.Flatten(start_dim=-2, end_dim=-1)
        # self.fc = nn.Linear(nb_filters, class_num)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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

        output = torch.cat(final_skip_connection, dim=-2)
        # output = self.weight_layer(output)
        output = torch.sum(torch.mul(output, self.weight_layer), dim=1)
        # output = self.flatten_2(output)
        # output = self.fc(output)
        # x = self.softmax(x)
        return output


class TIMNETClassification(nn.Module):
    def __init__(
        self,
        class_num,
        nb_filters=39,
        kernel_size=2,
        nb_stacks=1,
        dilations=8,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.TIMNET = TIMNET(
            class_num=class_num,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            dropout_rate=dropout_rate,
        )
        self.FC = nn.Linear(nb_filters, class_num)

    def forward(self, x):
        output = self.TIMNET(x)
        output = self.FC(output)
        return output

    def get_name(self):
        return "timnet"

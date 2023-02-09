import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            out_channels: int,
            kernel_sizes: list,
            output_dim: int,
            dropout=0.5,
    ):
        super().__init__()
        self.conv_0 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=out_channels,
                                kernel_size=kernel_sizes[0])

        self.conv_1 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=out_channels,
                                kernel_size=kernel_sizes[1])

        self.conv_2 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=out_channels,
                                kernel_size=kernel_sizes[2])

        self.fc = nn.Linear(len(kernel_sizes) * out_channels, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        embedded = text.permute(0, 2, 1)

        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        return self.fc(cat)

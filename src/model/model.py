import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            in_channels: int,
            out_channels: int,
            kernel_sizes: list,
            output_dim: int,
            dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, embedding_dim)
                )
            )

        self.fc = nn.Linear(len(kernel_sizes) * out_channels, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch_size, seq_len]
        embedded = self.embedding(x)

        # embedded = [batch_size, sent_len, emb_dim]
        embedded = embedded.unsqueeze(1)

        # embedded = [batch_size, 1, sent len, emb dim]
        many_conved = []
        for conv in self.convs:
            many_conved.append(
                F.leaky_relu(self.dropout(conv(embedded)).squeeze(3))
            )

        # conv_n = [batch_size, out_channels, sent_len - kernel_sizes[n]]
        pooled = []
        for conved in many_conved:
            pooled.append(
                F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            )

        # pooled_n = [batch_size, out_channels]
        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch_size, out_channels * len(kernel_sizes)]
        return self.fc(cat)

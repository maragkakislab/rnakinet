import torch

class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)


class RNNPooling(torch.nn.Module):
    def forward(self, x):
        output = x[0]  # All states
        h_n = x[1]  # Last hidden states
        max_pooled, _ = torch.max(output, dim=1)
        mean_pooled = torch.mean(output, dim=1)
        last_states = h_n.permute(1, 0, 2).flatten(start_dim=1)
        concat = torch.cat(
            [max_pooled, mean_pooled, last_states], dim=1)
        return concat


class RNAkinet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            
            
            Permute(),
            torch.nn.GRU(input_size=128, hidden_size=32, num_layers=1,
                         batch_first=True, bidirectional=True, dropout=0.0),
            RNNPooling(),
            torch.nn.Linear(1*(2*3*32), 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),

        )

    def forward(self, x):
        out = self.head(x)
        out = torch.sigmoid(out)
        return out

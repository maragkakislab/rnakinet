import torch
import math
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_layers=5, num_channels_initial=8, kernel_size=3, dilation=1, padding=0):
        super().__init__()
        
        layers = []
        in_channels = 1
        
        for i in range(num_layers):
            out_channels = num_channels_initial * (2 ** i)
            block = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=3),
            )
            layers.append(block)
            in_channels = out_channels
            
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    
class Permute(torch.nn.Module):
    #Putting the CNN output from bsXchannelsXlen to to batch_sizeXlenXchannels
    def forward(self, x):
        return x.permute(0, 2, 1)  
    
    
class RNNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, bidirectional=True, dropout=dropout),
            RNNPooling(),
        )
        
    def forward(self, x):
        return self.layers(x)
    
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

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        
    def forward(self, x):
        return self.layers(x)

    
class Attention(torch.nn.Module):
    def __init__(self, input_dim, len_limit):
        super(Attention, self).__init__()
        self.attention = torch.nn.Linear(input_dim, 1)

        # Initialize the position encoding matrix
        pe_matrix = torch.zeros(len_limit, input_dim)

        # Compute the position indices
        position = torch.arange(0, len_limit).unsqueeze(1)

        # Compute the dimension indices
        div_term = torch.exp(torch.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim))

        # Compute the sinusoidal encoding
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        # Register the position encoding matrix as a buffer to be saved with the model
        self.register_buffer('pe_matrix', pe_matrix)
        
    def forward(self, x):
        # Positional encoding
        x = x + self.pe_matrix[:x.size(1), :]
        # Compute attention scores
        attention_scores = self.attention(x).squeeze(-1)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)

        # Apply attention weights to the input sequence
        context_vector = torch.sum(x * attention_weights, dim=-2)

        return context_vector

    
    
    
#TODO LEGACY CLASS, REMOVE 
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
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
        )
        
    def forward(self, x):
        return self.layers(x)
    
    

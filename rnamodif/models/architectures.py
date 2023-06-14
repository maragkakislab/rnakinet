from types import SimpleNamespace
from RODAN.basecall import load_model
from rnamodif.models.generic import GenericUnlimited
import torch
# from rnamodif.models.modules import ConvNet, RNNEncoder, MLP, Attention, Permute, BigConvNet, ResConvNet
from rnamodif.models.modules import SimpleCNN, Permute, RNNEncoder, MLP , Attention

#TODO add dropout to cnns?
class CNN_RNN(GenericUnlimited):
    def __init__(self, 
                 cnn_depth, 
                 initial_channels=8, 
                 rnn_hidden_size=32,  
                 rnn_depth=1,
                 rnn_dropout=0.2,
                 mlp_hidden_size=30,
                 dilation=1,
                 padding=0,
                 **kwargs): 
        super().__init__(**kwargs)
        self.architecture = torch.nn.Sequential(
            SimpleCNN(num_layers=cnn_depth, dilation=dilation, padding=padding),
            Permute(),
            RNNEncoder(
                input_size=initial_channels*(2**(cnn_depth-1)), 
                hidden_size=rnn_hidden_size, 
                num_layers=rnn_depth, 
                dropout=rnn_dropout
            ),
            #biridectional (2x) and pooling is mean+max+last (3x)
            MLP(input_size=2*3*rnn_hidden_size, hidden_size=mlp_hidden_size),
        )
        
#TODO try dilation
class CNN_MAX(GenericUnlimited):
    def __init__(self, 
                 cnn_depth, 
                 initial_channels=8, 
                 mlp_hidden_size=30,
                 dilation=1,
                 padding=0,
                 **kwargs): 
        super().__init__(**kwargs)
        self.architecture = torch.nn.Sequential(
            SimpleCNN(num_layers=cnn_depth, dilation=dilation, padding=padding),
            torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(),
            MLP(initial_channels*(2**(cnn_depth-1)), hidden_size=mlp_hidden_size),
        )  

        
class CNN_ATT(GenericUnlimited):
    def __init__(self, 
                 cnn_depth, 
                 initial_channels=8, 
                 mlp_hidden_size=30,
                 dilation=1,
                 padding=0,
                 len_limit=400000,
                 **kwargs): 
        super().__init__(**kwargs)
        self.architecture = torch.nn.Sequential(
            SimpleCNN(num_layers=cnn_depth, dilation=dilation, padding=padding),
            Permute(),
            Attention(input_dim=initial_channels*(2**(cnn_depth-1)), len_limit=len_limit),
            MLP(initial_channels*(2**(cnn_depth-1)), hidden_size=mlp_hidden_size),
        )  
        
#TODO RODAN, RODAN embeddings+CNN 
        
# def get_rodan_model():
#     torchdict = torch.load(
#         '/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
#     origconfig = torchdict["config"]
#     args = {
#         'debug': False,
#         'arch': None,
#     }
#     trainable_rodan, device = load_model(
#         '/home/jovyan/RNAModif/RODAN/rna.torch', 
#         config=SimpleNamespace(**origconfig), 
#         args=SimpleNamespace(**args)
#     )
#     head = torch.nn.Sequential(
#         Permute(),
#         Attention(768, len_limit=len_limit),
#         torch.nn.Linear(768,1)
#     )
#     return torch.nn.Sequential(
#         trainable_rodan,
#         head
#     )


      
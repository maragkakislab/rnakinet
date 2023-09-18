from types import SimpleNamespace
# from RODAN.basecall import load_model
from rnamodif.models.generic import GenericUnlimited
import torch
# from rnamodif.models.modules import ConvNet, RNNEncoder, MLP, Attention, Permute, BigConvNet, ResConvNet
from rnamodif.models.modules import SimpleCNN, Permute, RNNEncoder, MLP , Attention
import torch.nn as nn
# from RODAN.basecall import load_model
# from RODAN.model import Mish
from types import SimpleNamespace

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
        self.save_hyperparameters() #For checkpoint loading
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
        self.save_hyperparameters() #For checkpoint loading
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
        self.save_hyperparameters() #For checkpoint loading
        self.architecture = torch.nn.Sequential(
            SimpleCNN(num_layers=cnn_depth, dilation=dilation, padding=padding),
            Permute(),
            Attention(input_dim=initial_channels*(2**(cnn_depth-1)), len_limit=len_limit),
            MLP(initial_channels*(2**(cnn_depth-1)), hidden_size=mlp_hidden_size),
        )  
        
#TODO pure RODAN model
        
# def get_rodan_model():
#     torchdict = torch.load(
#         '/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
#     origconfig = torchdict["config"]
#     args = {
#         'debug': False,
#         'arch': None,
#     }
#     #TODO check if im loading it ok
#     trainable_rodan, device = load_model(
#         '/home/jovyan/RNAModif/RODAN/rna.torch', 
#         config=SimpleNamespace(**origconfig), 
#         args=SimpleNamespace(**args)
#     )
#     return trainable_rodan

# def get_subrodan(depth=4, random_init=True):
#     model = get_rodan_model()
#     ch = list(model.named_children())
#     convlayers = ch[0][1]
#     submodule = convlayers[:depth]
#     if(random_init):
#         # print('randomizing subrodan')
#         submodule = submodule.apply(weight_reset)
#     return submodule

# def weight_reset(m):
#     for layer in m.children():
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()  

# class HYBRID(GenericUnlimited):
#     def __init__(self, 
#                  mlp_hidden_size,
#                  cnn_depth,
#                  initial_channels,
#                  subrodan_depth,
#                  subrodan_channels,
#                  kernel_size,
#                  **kwargs): 
#         super().__init__(**kwargs)
#         self.save_hyperparameters() #For checkpoint loading
#         #TODO does rodan turn off/on layers for inference - check
        
#         # rodan_channels = 5
#         rodan_channels = 768
        
#         channels = subrodan_channels+rodan_channels
        
#         class HYB_CNN(torch.nn.Module):
#             def __init__(self, num_layers, in_channels, init_channels=8, kernel_size=3, dilation=1, padding=0):
#                 super().__init__()

#                 layers = []
#                 for i in range(num_layers):
#                     out_channels = init_channels * (2 ** i)
#                     # out_channels = in_channels

#                     block = torch.nn.Sequential(
#                         torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding),
#                         torch.nn.ReLU(),
#                         torch.nn.MaxPool1d(kernel_size=3),
#                     )
#                     layers.append(block)
#                     in_channels = out_channels

#                 self.layers = torch.nn.Sequential(*layers)

#             def forward(self, x):
#                 return self.layers(x)

#         class merge_network(nn.Module):
#             def __init__(self, rodan_model, subrodan_model):
#                 super().__init__()
                
#                 self.rodan_model = rodan_model.convlayers
#                 # self.rodan_model = rodan_model
                
#                 for param in self.rodan_model.parameters():
#                     param.requires_grad = False
#                 for name, module in self.rodan_model.named_modules():
#                     if ('drop' in name):
#                         module.eval()
                    
#                 self.subrodan_model = subrodan_model
#                 self.dropout = torch.nn.Dropout(p=0.5)
                
            
#             def forward(self, x):
#                 subrodan_embedding = self.subrodan_model(x)
#                 # rodan_embedding = self.rodan_model(x).permute(1,2,0)
#                 rodan_embedding = self.rodan_model(x)
                
#                 subrodan_output_len = subrodan_embedding.shape[-1]
#                 rodan_output_len = rodan_embedding.shape[-1] 
#                 pad = rodan_output_len - subrodan_output_len
#                 rodan_embedding_cropped = rodan_embedding[:,:,0+pad//2:rodan_output_len-pad//2-pad%2]
#                 feature_concat = torch.cat([subrodan_embedding,rodan_embedding_cropped],1) 
#                 feature_concat = self.dropout(feature_concat)
                
#                 return feature_concat
                
#         self.architecture = torch.nn.Sequential(
#             #subrodan depth can be increased - up to ~21 to get more parameters for the base model
#             merge_network(rodan_model=get_rodan_model(), subrodan_model=get_subrodan(depth=subrodan_depth)),
#             HYB_CNN(num_layers=cnn_depth, in_channels=channels, init_channels=initial_channels, kernel_size=kernel_size),
#             torch.nn.AdaptiveMaxPool1d(1),
#             torch.nn.Flatten(),
#             MLP(initial_channels*(2**(cnn_depth-1)), hidden_size=mlp_hidden_size),
#         ) 
        
# #Taken from rodan
# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x *( torch.tanh(torch.nn.functional.softplus(x)))
            
# class RODANlike(GenericUnlimited):
#     def __init__(self, **kwargs): 
#         super().__init__(**kwargs)
#         self.save_hyperparameters() #For checkpoint loading
        
#         rodan_encoder=get_rodan_model().convlayers
#         self.architecture = torch.nn.Sequential(
#             rodan_encoder,
#             Permute(),
#             torch.nn.Linear(768,30),
#             Mish(),
#             torch.nn.Linear(30,1),
#             torch.nn.Flatten(),
#             torch.nn.AdaptiveMaxPool1d(1),
#         ) 
        
        
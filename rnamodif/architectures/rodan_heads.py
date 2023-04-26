import torch

class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x *( torch.tanh(torch.nn.functional.softplus(x)))

# light
def sotfmax_linear(input_size): #5
    return torch.nn.Sequential(
        torch.nn.Softmax(dim=-1),
        torch.nn.Linear(input_size,1),
        Permute((1,2,0)),
    )

# light conv
def softmax_conv(input_size, kernel_size): #5,5 (norm), 10 (wide)
    return torch.nn.Sequential(
        torch.nn.Softmax(dim=-1),
        Permute((1,2,0)),
        torch.nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=kernel_size),
    )


#light dense
def mish_dense(input_size):
    return torch.nn.Sequential(
        Mish(),
        torch.nn.Linear(input_size,100),
        Mish(),
        torch.nn.Linear(100,1),
        Permute((1,2,0)),
    )
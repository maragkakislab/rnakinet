from rnamodif.models.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.model_mine import MyModel

arch_map = {
    'rodan':RodanPretrainedUnlimited, 
    'cnn_gru':MyModel, 
}




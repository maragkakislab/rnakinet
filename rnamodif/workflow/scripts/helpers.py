from rnamodif.models.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.small_cnn import Small_CNN
from rnamodif.models.model_mine import MyModel
from rnamodif.models.architectures import CNN_RNN, CNN_MAX, CNN_ATT

arch_map = {
     #TODO deprecate non-generic networks 
    'custom_cnn': Small_CNN,
    'rodan':RodanPretrainedUnlimited,
    'cnn_gru':MyModel, #TODO remove, obsolete (After nanoid diffexp run)
    
    'cnn_rnn':CNN_RNN,
    'cnn_max':CNN_MAX,
    'cnn_att':CNN_ATT,
}
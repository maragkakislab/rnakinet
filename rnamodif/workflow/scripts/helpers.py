from rnamodif.models.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.small_cnn import Small_CNN
from rnamodif.models.model_mine import MyModel
from rnamodif.models.architectures import CNN_RNN, CNN_MAX, CNN_ATT #RODANlike #HYBRID #RODANlike

arch_map = {
     #TODO deprecate non-generic networks 
    'custom_cnn': Small_CNN,
    'rodan':RodanPretrainedUnlimited, #TODO uncomment and fix imports for reproducibility of rodan model
    'cnn_gru':MyModel, #TODO remove, obsolete (After nanoid diffexp run)
    
    'cnn_rnn':CNN_RNN,
    'cnn_max':CNN_MAX,
    'cnn_att':CNN_ATT,
    
    # 'hybrid':HYBRID,
    # 'RODANlike':RODANlike,
}




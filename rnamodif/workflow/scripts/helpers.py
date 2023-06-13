from rnamodif.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.small_cnn import Small_CNN
from rnamodif.model_mine import MyModel
arch_map = {
    'custom_cnn':Small_CNN,
    'rodan':RodanPretrainedUnlimited,
    'cnn_gru':MyModel,
}
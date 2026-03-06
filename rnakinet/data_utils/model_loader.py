# rename to model_loader.py, move to data_utils/
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_experimental import RNAkinet_LastOnly
from rnakinet.models.model_r9 import RNAkinet as RNAkinet_r9

arch_map = {
    'rnakinet':RNAkinet,
    'rnakinet_lastonly': RNAkinet_LastOnly,
    'rnakinet_r9': RNAkinet_r9,
}
default_models = {
    'rnakinet_r10_5EU': {
        'path':'checkpoints_pl/Human_batch_64_uniform/Human_batch_64_uniform/best-step=3520000-valid_loss=0.35.ckpt',
        'arch':'rnakinet',
        'unpadded': False
    },
    'rnakinet_r9_5EU' : {
        'path':'models/rnakinet_r9.ckpt',
        'arch':'rnakinet_r9',
        'unpadded': True
    }
}
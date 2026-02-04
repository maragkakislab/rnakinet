from rnakinet.models.model import RNAkinet
from rnakinet.models.model_experimental import RNAkinet_LastOnly

arch_map = {
    'rnakinet':RNAkinet,
    'rnakinet_lastonly': RNAkinet_LastOnly,
}
default_models = {
    'rnakinet_r10': {
        'path':'checkpoints_pl/Human_batch_64_uniform/Human_batch_64_uniform/best-step=3520000-valid_loss=0.35.ckpt',
        'arch':'rnakinet',
    },
}
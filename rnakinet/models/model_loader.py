# rename to model_loader.py, move to data_utils/
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_legacy import RNAkinet as RNAkinet_legacy

arch_map = {
    'v2.0': RNAkinet,
    'v1.0': RNAkinet_legacy,
}

default_models = {
    'r9_5EU_v1.0': {
        'tip': 'R9 model from RNAkinet v1, unpadded reads + v1.0 architecture',
        'path': 'models/rnakinet_r9_5EU_unpad@v1.0.ckpt',
        'arch': 'v1.0',
        'pad_reads': False,
    },
    'r10_5EU_v1.0': {
        'tip': 'R10 model from RNAkinet v1, unpadded reads + v1.0 architecture',
        'path': 'models/rnakinet_r10_5EU_unpad@v1.0',
        'arch': 'v1.0',
        'pad_reads': False,
    },
    'r10_5EU_v2.0': {
        'tip': 'R10 model from RNAkinet v2, padded reads + v2.0 architecture',
        'path': 'models/rnakinet_r10_5EU_pad@v2.0.ckpt',
        'arch': 'v2.0',
        'pad_reads': True,
    }      
}

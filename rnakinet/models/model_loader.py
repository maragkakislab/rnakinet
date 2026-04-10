# rename to model_loader.py, move to data_utils/
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_legacy import RNAkinet as RNAkinet_legacy

arch_map = {
    'v2.0': RNAkinet,
    'v1.0': RNAkinet_legacy,
}

default_models = {
    'r9_5EU_v1.0': {
        'tip': 'for ONT R9 kit with RNAkinet v1 architecture',
        'path': 'models/rnakinet_r9_5EU_unpad@v1.0.ckpt',
        'arch': 'v1.0',
        'pad_reads': False,
    },
    'r10_5EU_v1.0': {
        'tip': 'for ONT R10 kit with RNAkinet v1 architecture',
        'path': 'models/rnakinet_r10_5EU_unpad@v1.0',
        'arch': 'v1.0',
        'pad_reads': False,
    },
    'r10_5EU_v2.0': {
        'tip': 'for ONT R10 kit with RNAkinet v2 architecture',
        'path': 'models/rnakinet_r10_5EU@v2.0.ckpt',
        'arch': 'v2.0',
        'pad_reads': True,
    }      
}

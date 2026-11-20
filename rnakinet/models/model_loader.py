# rename to model_loader.py, move to data_utils/
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_legacy import RNAkinet as RNAkinet_legacy

arch_map = {
    'v2.0': RNAkinet,
    'v1.0': RNAkinet_legacy,
}

# whether inference pads reads to --max-len: v1.0 models were trained on
# unpadded reads, v2.0 models on padded reads
arch_pad_reads = {
    'v2.0': True,
    'v1.0': False,
}

# modification: the nucleoside analog the model detects (5EU or 4SU; 4SU
# models also apply to 4TU)
default_models = {
    'r9_5EU_v1.0': {
        'tip': '5EU, ONT R9 kit, RNAkinet v1 architecture',
        'modification': '5EU',
        'path': 'models/rnakinet_r9_5EU_unpad@v1.0.ckpt',
        'arch': 'v1.0',
    },
    'r10_5EU_v2.0': {
        'tip': '5EU, ONT R10 kit, RNAkinet v2 architecture',
        'modification': '5EU',
        'path': 'models/rnakinet_r10_5EU@v2.0.ckpt',
        'arch': 'v2.0',
    },
}

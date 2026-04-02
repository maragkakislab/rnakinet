# rename to model_loader.py, move to data_utils/
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_experimental import RNAkinet_LastOnly
from rnakinet.models.model_r9 import RNAkinet as RNAkinet_r9

arch_map = {'rnakinet': RNAkinet,
            'rnakinet_lastonly': RNAkinet_LastOnly,
            'rnakinet_r9': RNAkinet_r9
            }

default_models = {
    'rnakinet_r9_5EU': {
        # R9 model from RNAkinet v1, unpadded architecture
        'path': 'models/rnakinet_r9_5EU_unpad@v1.0.ckpt',
        'arch': 'rnakinet_r9',
        'pad_reads': False,
    },
    'rnakinet_r10_5EU': {
        # R10 model from RNAkinet v1, unpadded architecture
        'path': 'models/rnakinet_r10_5EU_unpad@v1.0',
        'arch': 'rnakinet',
        'pad_reads': False,
    },
}

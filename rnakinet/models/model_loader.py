import importlib.util
import os

from rnakinet.models.model import RNAkinet
from rnakinet.models.model_legacy import RNAkinet as RNAkinet_legacy
from rnakinet.models.model_v3 import RNAkinetV3

arch_map = {
    'v3.0': RNAkinetV3,
    'v2.0': RNAkinet,
    'v1.0': RNAkinet_legacy,
}


def get_arch(arch):
    """Model class of a packaged architecture name (arch_map), or of
    '<file.py>:<class>' for an architecture outside the package (e.g. an
    experimental one)."""
    if arch in arch_map:
        return arch_map[arch]
    path, sep, name = arch.rpartition(':')
    if not sep or not path.endswith('.py') or not os.path.isfile(path):
        raise ValueError(f"unknown architecture {arch!r}: use one of "
                         f"{list(arch_map)} or '<file.py>:<class>'")
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def input_mode(arch_class):
    """How an architecture takes reads, set by its class (input_mode):
    'padded' (zero-padded to --max-len; v2.0), 'unpadded' (one read at a
    time; v1.0) or 'variable' (padded to the longest read of each batch,
    with the reads' lengths passed to the model; v3.0)."""
    return getattr(arch_class, 'input_mode', 'padded')


def inference_lengths(arch_class):
    """An architecture's default inference --max-len (0: no limit) and
    --crop (0: whole reads): v3.0 scores reads of any length, cut to their
    first 200k samples; the others drop reads longer than 400k samples."""
    return (getattr(arch_class, 'inference_max_len', 400000),
            getattr(arch_class, 'inference_crop', 0))

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

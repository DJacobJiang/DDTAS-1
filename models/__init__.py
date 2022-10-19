# from .BN_Inception import BN_Inception
# from .meta_BN_Inception import meta_bn_inceptionv2, BN_Inception

from .BN_Inception import BN_Inception
from .BN_Inception_MetaAda import BN_Inception_MetaAda
from .ResNet_meta import meta_resnet50
from .ResNet import resnet50
from .BN_Inception_Attition import BN_Inception_At
from .R50_torchv import Resnet50_PA

__factory = {
    'BN_Inception': BN_Inception,
    'BN_Inception_MetaAda':BN_Inception_MetaAda,
    'ResNet50':resnet50,
    'ResNet50_MetaAda':meta_resnet50,
    'BNAt_Inception': BN_Inception_At,
    'ResNet50_P': Resnet50_PA
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    # print(name)
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)

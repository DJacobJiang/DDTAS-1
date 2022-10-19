from __future__ import print_function, absolute_import
from .Contrastive import ContrastiveLoss
from .MS import MSLoss
from .SoftCon import SoftContrastiveLoss

from .MS_ini import MS_ini
from .MS_W import Weight_MS as W_MS
from .Global_MS import Global_MS_m

from .Contrastive_ini import Contrastive_ini
from .Contrastive_W import ContrastiveW
from .Global_Contrastive import GlobalContrastive
from .triplet import TripletLoss

from .Binomial import BinomialLoss

__factory = {
    'Con': ContrastiveLoss,
    'MS': MSLoss,
    'SoftCon': SoftContrastiveLoss ,

    'SMS': MS_ini,
    'GMS': Global_MS_m,
    'WMS': W_MS,

    'Binomial': BinomialLoss,

    'Tri' : TripletLoss
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
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



# def names():
#     return sorted(__factory.keys())
#
#
# def create(name, *args, **kwargs):
#     """
#     Create a loss instance.
#
#     Parameters
#     ----------
#     name : str
#         the name of loss function
#     """
#     if name not in __factory:
#         raise KeyError("Unknown loss:", name)
#     return __factory[name]( *args, **kwargs)

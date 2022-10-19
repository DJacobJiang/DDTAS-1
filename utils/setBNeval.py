# Batch Norm Freezer : bring 2% improvement on CUB
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
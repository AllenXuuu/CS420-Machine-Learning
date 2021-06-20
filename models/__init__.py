from .UNet import Network as UNet
from .UNetPlus import Network as UNetPlus
from .AttentionUNet import Network as AttentionUNet
from .AttentionUNetPlus import Network as AttentionUNetPlus


def build_model(args, input_channel, n_class):
    if args.network.lower() == 'unet':
        return UNet(args, input_channel, n_class)
    elif args.network.lower() == 'unetplus':
        return UNetPlus(args, input_channel, n_class)
    elif args.network.lower() == 'attentionunet':
        return AttentionUNet(args, input_channel, n_class)
    elif args.network.lower() == 'attentionunetplus':
        return AttentionUNetPlus(args, input_channel, n_class)
    else:
        raise NotImplementedError

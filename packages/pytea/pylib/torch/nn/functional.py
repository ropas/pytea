import LibCall
from ..tensor import Tensor
import torch


def conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return LibCall.torch.conv2d(input, weight, bias, stride, padding, dilation, groups)


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    return LibCall.torch.conv_transpose2d(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )


def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
):
    # TODO: implement return_indices parameter
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return LibCall.torch.pool2d(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    return LibCall.torch.pool2d(input, kernel_size, stride, padding, 1, ceil_mode)


def nll_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
):
    return LibCall.torch.cross_entropy(input, target, not (reduction == "none"))


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
):
    return LibCall.torch.cross_entropy(input, target, not (reduction == "none"))


def binary_cross_entropy(
    input, target, weight=None, size_average=None, reduce=None, reduction="mean"
):
    result = LibCall.torch.sameShape(input, target)
    if reduction == "none":
        return result
    else:
        return LibCall.torch.scalarTensor()


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    return LibCall.torch.cosine_similarity(x1, x2, dim)


def linear(input, weight, bias=None):
    if bias is None:
        return input.matmul(weight.transpose(0, 1))
    else:
        return input.matmul(weight.transpose(0, 1)) + bias


def instance_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
):
    return LibCall.torch.identityShape(input)


# pad should be even-length tuple or list
def pad(input, pad, mode="constant", value=0):
    return LibCall.torch.pad(input, pad)


def mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    result = LibCall.torch.sameShape(input, target)

    if reduction == "none":
        return result
    else:
        return torch.Tensor()


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return LibCall.torch.identityShape(input)


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return LibCall.torch.identityShape(input)


def dropout(input, p=0.5, training=True, inplace=False):
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got ...")
    return LibCall.torch.identityShape(input)


def relu(input):
    return LibCall.torch.identityShape(input)


def gelu(input):
    return LibCall.torch.identityShape(input)


def tanh(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def sigmoid(input, out=None):
    result = LibCall.torch.identityShape(input)
    LibCall.torch.copyOut(result, out)
    return result


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # TODO: implement weight and bias
    norm_tensor = Tensor(*normalized_shape)
    return LibCall.torch.layer_norm(input, norm_tensor, weight, bias)


def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    return LibCall.torch.interpolate(input, size, scale_factor)


def pixel_shuffle(input, upscale_factor):
    return LibCall.torch.pixel_shuffle(input, upscale_factor)

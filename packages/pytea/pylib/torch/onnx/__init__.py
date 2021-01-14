import LibCall


def _export(*args, **kwargs):
    return LibCall.builtins.warn("torch.onnx._export will return unknown value.")


def export(*args, **kwargs):
    return LibCall.builtins.warn("torch.onnx._export will return unknown value.")

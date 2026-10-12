"""Local precision controls shared by full-training and evaluation entrypoints."""
from __future__ import annotations


def configure_model_precision(model):
    """Force only the first base-model convolution to execute without TF32.

    The wrapper is an instance-level forward policy: it adds no module,
    parameter, or buffer and therefore cannot alter state_dict compatibility.
    Repeated calls are idempotent.
    """
    import torch

    try:
        stem = model.base_model.model[0].conv
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("model lacks expected base_model.model[0].conv stem") from exc
    if getattr(stem, "_full_training_stem_tf32_disabled", False):
        return model
    original_forward = stem.forward

    def forward_without_tf32(*args, **kwargs):
        # torch.backends.cudnn.flags defaults enabled/deterministic to False;
        # carry every surrounding execution-policy flag through explicitly so
        # this wrapper changes TF32 only and continues to use cuDNN.
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=False,
        ):
            return original_forward(*args, **kwargs)

    stem.forward = forward_without_tf32
    stem._full_training_stem_tf32_disabled = True
    return model

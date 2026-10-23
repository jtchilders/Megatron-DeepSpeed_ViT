# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
from deepspeed.accelerator.real_accelerator import get_accelerator

if get_accelerator().device_name() == 'xpu':
    # IPEX was a mandatory separate package on older Aurora stacks. On the
    # current frameworks module its XPU support is upstreamed into torch
    # (torch.xpu), and the package is no longer installed, so importing it
    # is best-effort. The IPEX-only ops this file reaches for are probed
    # through torch.xpu below.
    try:
        import intel_extension_for_pytorch
    except ImportError:
        pass
if get_accelerator().device_name() == 'cuda':
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
    from apex.normalization import MixedFusedRMSNorm as RMSNorm
else:
    if hasattr(torch.xpu, "IpexRmsNorm"):
        from .fused_rmsnorm import RMSNorm
    else:
        from .rmsnorm import RMSNorm
    from torch.nn import LayerNorm

from .distributed import DistributedDataParallel
from .bert_model import BertModel
from .gpt_model import GPTModel, GPTModelPipe
from .t5_model import T5Model
from .language_model import get_language_model
from .module import Float16Module

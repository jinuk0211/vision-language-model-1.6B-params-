import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,  

# is_flash_attn_2_available() 함수는 챗봇 모델에서 Flash Attention 2 (FA2) 기능을 사용할 수 있는지 확인하는 데 사용됩니다.
# FA2는 챗봇 모델의 성능을 향상시키는 데 도움이 되는 새로운 기술입니다

    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration import phiconfig

try:  
    if is_flash_attn_2_available(): #flash attention 사용할수 있으면 사용하자
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    # Workaround for https://github.com/huggingface/transformers/issues/28459,
    # don't move to contextlib.suppress(ImportError)
    pass

logger = logging.get_logger(__name__)

class rotaryembedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # torch.jit.trace 가 가능하게 만들자
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):

class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
  
def rotate_half(x):

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):

class phiMLP(nn.Module):

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

class phiAttention(nn.Module):

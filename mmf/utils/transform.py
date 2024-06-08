# Copyright (c) Facebook, Inc. and its affiliates.

# Last Change:  2024-05-07 10:44:38
from typing import Optional
import torch
from torch import Tensor


def transform_to_batch_sequence(tensor: Tensor) -> Tensor:
    if len(tensor.size()) == 2:
        return tensor
    else:
        assert len(tensor.size()) == 3
        return tensor.contiguous().view(-1, tensor.size(-1))


def transform_to_batch_sequence_dim(tensor: Tensor) -> Tensor:
    if len(tensor.size()) == 3:
        return tensor
    else:
        assert len(tensor.size()) == 4
        return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))

def prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    Mapping 1 to float 0, and 1 to float -inf
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    #return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -10000.0)

import torch
from torch.nn import functional


def apply_nd(fn, input):
    """
    Performs an operation which is defined only for a 2-D tensor.
    It assumes that only values in the last dimension affect the result.
    """

    original_size = input.size()
    input_flat = input.contiguous().view(-1, original_size[-1])
    output_flat = fn(input_flat)
    output_size = original_size[:-1] + (output_flat.size(-1),)
    output = output_flat.view(*output_size)
    return output


def sequence_mask(length, max_length=None):
    """
    Args:
        length (Tensor): A long tensor of size (batch_size,).
        max_length (int): The maximum length. If None, it automatically
            sets this as max(lengths).
    Returns:
        mask: (Tensor): A byte mask tensor of size
            (max_length, batch_size). Each element is 1 if valid
            and 0 else.
    """

    if max_length is None:
        max_length = length.max()
    seq_range = torch.arange(0, max_length).unsqueeze(1).long()
    if length.is_cuda:
        device = length.get_device()
        seq_range = seq_range.cuda(device)
    length = length.unsqueeze(0)
    mask = torch.lt(seq_range, length)
    return mask


def sequence_cross_entropy(logits, targets, length):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = functional.log_softmax(logits_flat)
    target_flat = targets.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*targets.size())
    mask = sequence_mask(length=length, max_length=logits.size(0))
    losses.data.masked_fill_(mask=~mask, value=0)
    loss = losses.sum() / losses.size(1)
    return loss

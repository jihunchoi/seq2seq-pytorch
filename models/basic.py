import torch
from torch.nn import functional


def sequence_mask(length, max_length=None):
    """
    Args:
        length (Tensor): A long tensor of size (batch_size,).
        max_length (int): The maximum length. If None, it automatically
            sets this as max(lengths).
    Returns:
        mask: (Tensor): A byte mask tensor of size
            (batch_size, max_length). Each element is 1 if valid
            and 0 else.
    """

    if max_length is None:
        max_length = length.max()
    seq_range = torch.arange(0, max_length).unsqueeze(0).type_as(length)
    length = length.unsqueeze(1)
    mask = torch.lt(seq_range, length)
    return mask


def sequence_cross_entropy(logits, targets, length):
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    losses_flat = functional.cross_entropy(
        input=logits_flat, target=targets_flat, reduce=False)
    losses = losses_flat.view(*targets.size())
    mask = sequence_mask(length=length, max_length=logits.size(0)).t()
    losses.data.masked_fill_(mask=~mask, value=0)
    loss = losses.sum() / losses.size(1)
    return loss

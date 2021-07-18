from torch import nn

def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()
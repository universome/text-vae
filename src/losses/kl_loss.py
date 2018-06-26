import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self, size_average=True):
        super(KLLoss, self).__init__()

        self.size_average = size_average

    def forward(self, means, stds):
        losses = kld_between_isotropic_and_standard(means, stds)
        loss = losses.mean() if self.size_average else losses.sum()

        return loss


def kld_between_isotropic_and_standard(means, log_stds):
    """
    Computes distance between isotropic normal and standard normal distributions
    """
    means_term = means.pow(2)
    log_stds_term = 2 * log_stds
    stds_term = log_stds.exp().pow(2)

    return 0.5 * (means_term + stds_term - log_stds_term - 1).sum(dim=1)

import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self, size_average=True):
        super(KLLoss, self).__init__()

        self.size_average = size_average

    def forward(self, means, stds):
        losses = kld_between_isotropic_and_standard(means, stds)
        loss = losses.mean() if self.size_average else losses.sum()

        return loss


def kld_between_isotropic_and_standard(means, stds):
    """
    Computes distance between isotropic normal and standard normal distributions
    """
    means_term = means.pow(2).sum(dim=1)
    stds_term = stds.pow(2).sum(dim=1)
    log_stds_term = stds.pow(2).log().sum(dim=1)
    n_dim = means.size(1)

    return 0.5 * (means_term + stds_term - log_stds_term - n_dim)

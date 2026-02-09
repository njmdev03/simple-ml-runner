import torch
from torch.utils.data import DataLoader

# Computes the mean and standard deviation of a dataset.
# Uses Welford's algorithm to avoid loading the entire dataset in memory.
def compute_stats(dataset, batch_size=1024):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    n_samples = 0
    mean = 0.0
    M2 = 0.0

    for x, _ in loader:
        x = x.float()

        # Determine axes for mean/std
        if x.dim() > 2:  # images
            axes = (0, 2, 3)
        else:  # tabular
            axes = (0,)

        batch_samples = x.size(0)
        batch_mean = x.mean(dim=axes)
        batch_var  = x.var(dim=axes, unbiased=False)

        if n_samples == 0:
            mean = batch_mean
            M2 = batch_var * batch_samples
        else:
            delta = batch_mean - mean
            total = n_samples + batch_samples
            mean += delta * batch_samples / total
            M2 += batch_var * batch_samples + delta**2 * n_samples * batch_samples / total

        n_samples += batch_samples

    variance = M2 / n_samples
    std = torch.sqrt(variance)
    return mean, std

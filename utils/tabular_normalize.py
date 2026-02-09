# A normalization method for use on tabular data
# Used by the UCI Adult Income example
class TabularNormalize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

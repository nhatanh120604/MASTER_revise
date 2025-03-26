import torch
import torch.nn as nn
import numpy as np

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log Likelihood loss for uncertainty estimation.
    This loss function expects model to output mean prediction and log variance.
    """
    def __init__(self, eps=1e-6, min_log_var=-20, max_log_var=20):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, mean, log_var, target):
        """
        Args:
            mean: predicted mean
            log_var: predicted log variance
            target: ground truth target
        """
        # Clip log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=self.min_log_var, max=self.max_log_var)

        # Convert log variance to variance
        var = torch.exp(log_var) + self.eps  # add epsilon for numerical stability

        # Calculate negative log likelihood
        loss = 0.5 * (log_var + ((target - mean) ** 2) / var)

        return loss.mean()

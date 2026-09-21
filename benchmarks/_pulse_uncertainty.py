"""Vendored from paper/pulse/sparc-nn/uncertainty_loss.py; MIT, Copyright (c) 2025 Han Bui.
See LICENSE-PULSE. Source implementation below is unchanged.
"""
import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_dict, priority_weights=None):
        total_loss = 0
        
        for i, (key, val) in enumerate(loss_dict.items()):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            uncertainty_term = 0.5 * precision * val + 0.5 * log_var
            
            user_w = 1.0
            if priority_weights and key in priority_weights:
                user_w = priority_weights[key]
            
            weighted_loss = user_w * uncertainty_term
            total_loss += weighted_loss
            
        return total_loss
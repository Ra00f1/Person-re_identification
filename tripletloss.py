import numpy as np
import torch
import torch.nn as nn

MARGIN = 0.3


class TripletLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # pos_dist = np.sum(np.square(anchor - positive))
        # neg_dist = np.sum(np.square(anchor - negative))

        # Calculate distances using L2 norm
        anchor_positive_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        anchor_negative_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        loss = anchor_positive_dist - anchor_negative_dist + self.margin
        return torch.clamp(loss, min=0)



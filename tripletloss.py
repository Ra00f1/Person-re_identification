import tensorflow as tf
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances and loss
        anchor_positive_dist = tf.norm(anchor - positive, axis=1)
        anchor_negative_dist = tf.norm(anchor - negative, axis=1)
        loss = anchor_positive_dist - anchor_negative_dist + self.margin
        loss = tf.maximum(loss, 0.0)

        return loss  # Return both loss and tape

    def get_config(self):
        config = super(TripletLoss, self).get_config()
        config.update({'margin': self.margin})
        return config
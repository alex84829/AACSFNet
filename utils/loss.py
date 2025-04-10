import torch
import torch.nn as nn
import numpy as np


class MSE_weighted(nn.Module):
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(MSE_weighted, self).__init__()
        self.weight_type = weight_type
        self.device = device

    def forward(self, pred, target):
        if self.weight_type == 'mean':
            len = target.size(1)
            self.weights = torch.ones((1, len)) / (len * 1.0)
            self.weights = self.weights.to(self.device)
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = torch.mean(((pred - target) ** 2) * self.weights)
        return loss

    def prepare_dynamic_weights(self, target):
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in target])
        weight = 1. / np.log2(class_sample_count + 1.2)
        weight = torch.from_numpy(weight)
        return weight.double()


if __name__ == '__main__':
    preds = torch.tensor([1.3, 2.5, 3.4, 5.1, 2.5])
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0, 2.0])
    loss = MSE_weighted('dynamic')(preds, labels)
    print(loss)
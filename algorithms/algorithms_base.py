"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

import torch
import torch.nn as nn


class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, data):
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            feat = self.feature_extractor(data)
            pred = self.classifier(feat)
        return pred
    
    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
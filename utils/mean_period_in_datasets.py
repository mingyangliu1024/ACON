import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.models import *
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from models.augmentations import jitter, scaling, permutation
from algorithms.algorithms_base import Algorithm


def mean_period_in_datasets(dataloader, k=2):
    list_a = []
    for data, label in iter(dataloader):
        a = torch.fft.rfft(data, dim=-1,norm='ortho').abs()
        list_a.append(a)
    # all_data_num, channel_num, time_steps/2

    a = torch.cat(list_a, dim=0)
    # time_steps/2
    a = a.mean(0).mean(0)
    a[0] = 0
    _, top_list = torch.topk(a, k)
    top_list = top_list.detach().cpu().numpy()
    #print(top_list)
    period = data.shape[-1] // top_list
    print(period)
    return period
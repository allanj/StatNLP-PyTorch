import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import GlobalNetworkParam


class NetworkModel(nn.Module):

    def __init__(self, fm, compiler):
        super(NetworkModel, self).__init__()
        self.gnp = GlobalNetworkParam()
        self.fm = fm
        self.compiler = compiler



    def finalize(self):
        self.gnp.lock_it()

    def forward(self, *input):
        pass


    def predict(self, input):
        pass
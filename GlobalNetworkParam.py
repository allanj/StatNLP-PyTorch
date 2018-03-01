import torch.nn
from Utils import *
from StringIndex import StringIndex


class GlobalNetworkParam:

    def __init__(self):
        self._string_index = StringIndex()
        self.locked = False
        self.feature_int_dict = {}
        self._size = 0
        self.weights = None
        self.version = 0

    def get_version(self):
        return self.version

    def to_int(self, s):
        return self._string_index.get_or_put(s)


    def to_feature(self, network, ftype, foutput, finput):
        fname = ftype + foutput + finput
        fname_id = network.param.to_int(fname) if self._string_index == None else self.to_int(fname)

        return self.to_feature_id(network, fname_id)

    def is_locked(self):
        return self.locked

    def get_weight(self, f):
        return self.weights[f]

    def to_feature_id(self, network, fname_id):

        if self.is_locked():
            if fname_id in self.feature_int_dict:
                return self.feature_int_dict[fname_id]
            else:
                return -1
        else:
            if fname_id in self.feature_int_dict:
                return self.feature_int_dict[fname_id]
            else:
                self.feature_int_dict[fname_id] = len(self.feature_int_dict)
                self._size += 1
                return self.feature_int_dict[fname_id]


    # def lock_it(self):
    #     if self.is_locked():
    #         return
    #
    #     weights_new = torch.nn.Parameter(torch.Tensor(self.size()).fill_(0.0)) #self.size
    #     print('weights_new type:', type(weights_new))
    #     #weights_new.fill_(0.0)  ## TODO: need to randomly initialize
    #     self.weights = weights_new
    #     self.version = 0
    #     self._string_index.lock()
    #     self.locked = True
    #     eprint(self._size, " features")


    def size(self):
        return self._size







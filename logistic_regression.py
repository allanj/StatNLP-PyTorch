import NetworkCompiler
import NetworkIDMapper
import numpy as np
from enum import Enum

class LRNetworkCompiler(NetworkCompiler):

    def __init__(self):
        super.__init__(LRNetworkCompiler)
        NetworkIDMapper.set_capacity(np.asarray([10]))

    NodeType = Enum(['X', 'N', 'Y', 'Root'])

    def get_node_type_size(self):
        return len(LRNetworkCompiler.NodeType)

    def to_node_Root(self):
        return NetworkIDMapper.toHybridNodeID([1000, LRNetworkCompiler.NodeType.Root]);

    def compile_labeled(self, network_id, inst, param):
        pass


    def compile_unlabeled(self, network_id, inst, param):
        pass


    def decompile(self, network):
        pass
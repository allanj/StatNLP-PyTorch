import NetworkCompiler
import BaseNetwork
import BaseNetwork.NetworkBuilder
import LinearInstance
import NetworkIDMapper
import numpy as np
from enum import Enum

class LinearCRFCompiler(NetworkCompiler):

    def __init__(self):
        super.__init__(LinearCRFCompiler)
        NetworkIDMapper.set_capacity(np.asarray([1000, 1000]))

    NodeType = Enum('X', 'O', 'I', 'B', 'Root')


    def get_node_type_size(self):
        return len(LinearCRFCompiler.NodeType)

    def to_node_Root(self):
        return NetworkIDMapper.toHybridNodeID([1000, LinearCRFCompiler.NodeType.Root]);



    def compile_labeled(self, network_id, inst : LinearInstance, param):
        '''
		int size = inst.size();
        :param network_id:
        :param inst:
        :param param:
        :return:
        '''
        crfnetwork = BaseNetwork.NetworkBuilder.builder()
        size = inst.size()

        crfnetwork.add_node()



    def compile_unlabeled(self, network_id, inst, param):
        pass

    def decompile(self, network):
        pass
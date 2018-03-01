from NetworkCompiler import NetworkCompiler
import numpy as np
from NetworkIDMapper import NetworkIDMapper
from BaseNetwork import BaseNetwork
from GlobalNetworkParam import GlobalNetworkParam
from Instance import Instance
from FeatureManager import FeatureManager
from FeatureArray import FeatureArray
from LinearInstance import LinearInstance
from NetworkModel import NetworkModel
from enum import Enum


class NodeType(Enum):
    LEAF = 0
    NODE = 1
    ROOT = 2


class LRInstance(Instance):
    def __init__(self, instance_id, weight, input=None, output=None):
        super.__init__(LinearInstance, instance_id, weight=1.0, input=None, output=None)



    def size(self):
        return len(input)

    def duplicate(self):
        dup = LinearInstance(self.instance_id, self.weight, self.input, self.output)
        return dup

    def removeOutput(self):
        self.output = None

    def removePrediction(self):
        self.prediction = None

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction

    def has_output(self):
        return self.output != None

    def has_prediction(self):
        return self.prediction != None


class LRFeatureManager(FeatureManager):
    def __init__(self, param_g):
        super.__init__(LRFeatureManager, param_g)

    def extract_helper(self, network, parent_k, children_k, children_k_index):
        parent_arr = network.get_node_array(parent_k)
        node_type_id = parent_arr[0]
        if node_type_id == 0 and node_type_id == 2:
            return FeatureArray.EMPTY
        inst = network.get_instance()
        ft_strs = inst.get_input().split(",")

        fs = []
        label_id = parent_arr[1]
        fs.append(self._param_g.toFeature(network, "location", label_id, ft_strs[0]))
        fs.append(self._param_g.toFeature(network, "quality", label_id, ft_strs[1]))
        fs.append(self._param_g.toFeature(network, "people", label_id, ft_strs[2]))

        return self.createFeatureArray(network, fs)


##Done
class LRReader():

    @staticmethod
    def read_insts(file, is_labeled, number):
        insts = []
        f = open(file, 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            fields = line.split(' ')
            inputs = fields[0]
            output = fields[1]
            inst = LRInstance(len(insts) + 1, 1, inputs, output)
            insts.append(inst)
        f.close()

        return insts


class LRNetworkCompiler(NetworkCompiler):

    def __init__(self):
        ##node type and label id
        NetworkIDMapper.set_capacity(np.asarray([3, 2]))
        self.build_generic_network()
        self._all_nodes = None
        self._all_children = None

    def to_leaf(self):
        return NetworkIDMapper.to_hybrid_node_ID(np.asanyarray([0, 0]))

    def to_node(self, label_id):
        return NetworkIDMapper.to_hybrid_node_ID(np.asanyarray([1, label_id]))

    def to_root(self):
        return NetworkIDMapper.to_hybrid_node_ID(np.asanyarray([2, 0]))

    def compile_labeled(self, network_id, inst, param):
        builder = BaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()

        node = self.to_node(inst.get_output)
        builder.add_node(node)
        builder.add_edge(node, [leaf])

        root = self.to_root()
        builder.add_node(root)
        builder.add_edge(root, [node])

        network = builder.build(network_id, inst, param, self)

        return network

    def compile_unlabled(self, network_id, inst, param):
        root = self.to_root()
        root_idx = self._all_nodes.index(root)
        num_nodes = root_idx + 1
        return BaseNetwork.NetworkBuilder.quick_build(network_id, inst, self._all_nodes, self._all_children, num_nodes,
                                                      param, self)

    def build_generic_network(self):
        BaseNetwork.NetworkBuilder
        builder = BaseNetwork.NetworkBuilder.builder()
        leaf = self.to_leaf()
        leaves = [leaf]
        builder.add_node(leaf)
        root = self.to_root()
        builder.add_node(root)
        for label_id in range(2):
            node = self.to_node(label_id)
            builder.add_node(node)
            builder.add_edge(node, leaves)
            builder.add_edge(root, [node])
        builder.build(None, None, None, None)
        self._all_nodes = builder.get_all_nodes()

        self._all_children = builder.get_all_children()


    def decompile(self, network):
        inst = network.get_instance()
        root = self.to_root()
        node_idx = self._all_nodes.index(root)
        label_node_idx = network.get_max_path(node_idx)[0]
        arr = network.get_node_array(label_node_idx)
        label_id = arr[1]
        inst.set_prediction(label_id)
        return inst


if __name__ == "__main__":
    train_insts = LRReader.read_inst("train.txt", True)

    gnp = GlobalNetworkParam()
    fm = LRFeatureManager(gnp)
    compiler = LRNetworkCompiler()

    model = NetworkModel(fm, compiler)
    model.train(train_insts, 1000)

    # test_insts = LRReader.read_inst(train_file, False)
    # results = model.test(test_insts)


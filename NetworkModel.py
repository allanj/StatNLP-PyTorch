import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from Utils import *
import GlobalNetworkParam
import NetworkConfig
import torch.optim


class NetworkModel(nn.Module):

    def __init__(self, fm, compiler):
        super.__init__(NetworkModel)
        self._fm = fm
        self._num_threads = NetworkConfig.NUM_THREADS
        self._compiler = compiler
        self._all_instances = None
        self._param = None
        self._cache_networks = True
        self._networks = None

    def get_instances(self):
        return self._all_instances

    def get_feature_manager(self):
        return self._fm

    def get_network_compiler(self):
        return self._compiler

    def split_instances_for_train(self):
        eprint("#instances=", len(self._all_instances))
        insts = [None for i in range(len(self._all_instances))]
        insts_list = [None for i in range(len(self._all_instances))]
        thread_id = 0
        for k in range(len(self._all_instances)):
            inst = self._all_instances[k]
            insts_list[thread_id].append(inst)
            thread_id = (thread_id + 1) % self._num_threads

        for thread_id in range(self._num_threads):
            size = len(insts_list[thread_id])
            insts[thread_id] = [None for i in range(2 * size)]
            for i in range(size):
                inst = insts_list[thread_id][i]
                insts[thread_id][2 * i + 1] = inst
                inst_new = inst.duplicate()
                inst_new.set_instance_id(-inst.get_instance_id())
                inst_new.set_weight(-inst.get_weight())
                inst_new.set_unlabeled()
                insts[thread_id][2 * i] = inst_new
            print("Thread ", thread_id, " has ", len(insts[thread_id]), " instances.")

        return insts

    def train(self, train_insts, max_iterations):
        insts = self.prepare_instance_for_compilation(train_insts)
        if NetworkConfig.PRE_COMPILE_NETWORKS:
            self.pre_compile_networks(insts)
        keep_existing_threads = True if NetworkConfig.PRE_COMPILE_NETWORKS else False
        self.touch(insts, keep_existing_threads)

        self._param.finalize_it()

        self._fm.get_param_g.lock_it()

        optimizer = torch.optim.LBFGS(self.parameters())  # lr=0.8
        for it in range(max_iterations):

            optimizer.zero_grad()

            all_loss = 0  ### scalar

            for i in range(len(self._all_instances)):
                loss = self.forward(self.get_network(self._all_instances[i]))
                all_loss += loss
                loss.backward()
            optimizer.step()


    def forward(self, network):
        return network.inside()



    def get_network(self, network_id):
        if self._cache_networks and self._networks[network_id] != None:
            return self._networks[network_id]

        network = self._compiler.compile(network_id, self._all_instances[network_id], self.param)

        if self._cache_networks:
            self._networks[network_id] = network

        return network


    def touch(self):
        if self._networks == None:
            self._networks = [None for i in range(len(self._all_instances))]

        for network_id in range(len(self._all_instances)):
            if network_id % 100 == 0:
                eprint('.', end='')
            self.get_network(network_id).touch()

        eprint()


    def test(self, instances):
        return self.decode(instances=instances)

    def decode(self, instances, cache_features=False):


        self._num_threads = NetworkConfig.NUM_THREADS
        eprint('#Threads: ', self._num_threads)

        self._all_instances = instances

        instances_output = []
        for k in range(len(instances)):
            instance = instances[k]
            network = self._compiler.compile(k, instance, self.param)
            network.max()
            instance_output = self._compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output





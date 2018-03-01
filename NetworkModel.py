import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from GlobalNetworkParam import GlobalNetworkParam
from NetworkConfig import NetworkConfig
from LocalNetworkParam import LocalNetworkParam
import torch.optim
from Utils import *



class NetworkModel(nn.Module):

    def __init__(self, fm, compiler):
        super().__init__()
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

    def split_instances_for_train(self, insts_before_split):
        eprint("#instances=", len(insts_before_split))
        insts = [None for i in range(len(insts_before_split) * 2)]

        k=0
        for i in range(0, len(insts), 2):
            insts[i + 1] = insts_before_split[k]
            insts[i] = insts_before_split[k].duplicate()
            insts[i].set_instance_id(-insts[i].get_instance_id())
            insts[i].set_weight(-insts[i].get_weight())
            insts[i].set_unlabeled()
            k = k + 1
        return insts

    def lock_it(self):
        gnp = self._fm.get_param_g()

        if gnp.is_locked():
            return

        # weights_new = torch.nn.Parameter(torch.randn(gnp.size()))  # self.size
        weights_new = torch.nn.Parameter(torch.Tensor(gnp.size()).fill_(0.0))  # self.size
        print('weights_new type:', type(weights_new))
        # weights_new.fill_(0.0)  ## TODO: need to randomly initialize


        self.weights = weights_new
        gnp.version = 0
        gnp._string_index.lock()
        gnp.locked = True
        eprint(gnp._size, " features")

    def train(self, train_insts, max_iterations):
        insts_before_split = train_insts #self.prepare_instance_for_compilation(train_insts)

        insts = self.split_instances_for_train(insts_before_split)

        self._param = LocalNetworkParam(self, self._fm, len(insts))
        self._fm.set_local_param(self._param)

        self._all_instances = insts
        if NetworkConfig.PRE_COMPILE_NETWORKS:
            self.pre_compile_networks(insts)
        keep_existing_threads = True if NetworkConfig.PRE_COMPILE_NETWORKS else False
        self.touch(insts, keep_existing_threads)


        self._param.finalize_it()

        #self._fm.get_param_g().lock_it()
        self.lock_it()


        # optimizer = torch.optim.SGD(self.parameters(), lr = 0.01)  # lr=0.8
        optimizer = torch.optim.LBFGS(self.parameters())  # lr=0.8
        iter = 0
        for it in range(max_iterations):

            def closure():

                optimizer.zero_grad()

                all_loss = 0  ### scalar

                for i in range(len(self._all_instances)):
                    loss = self.forward(self.get_network(i))
                    all_loss -= loss
                    #loss.backward()

                all_loss.backward()
                print("Iteration ", it,": Obj=",  all_loss.data[0])
                # iter += 1

                return all_loss
            #print('bWeight:', self.weights)
            # print("bGrad:", self.weights.grad)
            optimizer.step(closure)

            if iter > max_iterations:
                break
            #print('aWeight:', self.weights)
            # print("aGrad:", self.weights.grad)


    def forward(self, network):
        return network.inside()



    def get_network(self, network_id):
        if self._cache_networks and self._networks[network_id] != None:
            return self._networks[network_id]

        inst = self._all_instances[network_id]

        network = self._compiler.compile(network_id, inst, self._param)
        #print("after compile: ", network)


        if self._cache_networks:
            self._networks[network_id] = network

        return network


    def touch(self, insts, keep_existing_threads = False):
        if self._networks == None:
            self._networks = [None for i in range(len(insts))]

        for network_id in range(len(insts)):
            if network_id % 100 == 0:
                eprint('.', end='')
            network =  self.get_network(network_id)

            network.touch()

        eprint()


    def test(self, instances):
        return self.decode(instances=instances)

    def decode(self, instances, cache_features=False):


        self._num_threads = NetworkConfig.NUM_THREADS
        eprint('#Threads: ', self._num_threads)

        self._all_instances = instances

        self._param = LocalNetworkParam(self, self._fm, len(instances))
        self._fm.set_local_param(self._param)

        instances_output = []

        for k in range(len(instances)):
            instance = instances[k]
            # print("decode ", instance.is_labeled)

            network = self._compiler.compile(k, instance, self._param)
            network.max()
            instance_output = self._compiler.decompile(network)
            instances_output.append(instance_output)

        return instances_output





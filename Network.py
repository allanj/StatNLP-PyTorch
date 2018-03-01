from abc import ABC, abstractmethod
#import NetworIDMapper
from NetworkIDMapper import *
import torch
import math
import numpy as np
import torch.autograd as autograd
from Utils import *

class Network:

    def __init__(self, network_id, instance, param):
        self.network_id = network_id
        self.inst = instance
        self.param = param
        self.maxSharedArray = None
        self.maxPathsSharedArrays = None

    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst

    def getInsideSharedArray(self):
        if self.inside_shared_array == None or self.count_nodes() > len(self.inside_shared_array):
            self.inside_shared_array = autograd.Variable(
                torch.Tensor(self.count_nodes()).fill_(0.0))  ##np.zeros(self.count_nodes(), dtype=np.float)


        return self.inside_shared_array

    def inside(self):
        self.inside_scores = autograd.Variable(torch.Tensor(self.count_nodes()).fill_(0.0))
        for k in range(self.count_nodes()):
            self.get_inside(k)
        if math.isinf(self.get_insides().data[0]) and self.get_insides() > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        # if self.get_instance().is_labeled:
        #     print('inside:', self.get_insides())
            #print(self.param.model.weights)
        #
        #
        # print()
        #
        # print()
        weight = self.get_instance().weight
        return self.get_insides() * weight

    def get_insides(self):
        return self.inside_scores[self.count_nodes() - 1]

    def get_inside(self, k):
        if self.is_removed(k):
            self.inside_scores[k] = -math.inf
            return

        inside_score = -math.inf
        children_list_k = self.get_children(k)
        ## If this node has no child edge, assume there is one edge with no child node
        ## This is done so that every node is visited in the feature extraction step below
        if len(children_list_k) == 0:
            children_list_k = np.zeros((1, 0))


        # print('children_list_k:', children_list_k, ' k:', k, 'len(children_list_k):', len(children_list_k))

        scores = autograd.Variable(torch.Tensor(len(children_list_k)))
        for children_k_index in range(len(children_list_k)):
            children_k = children_list_k[children_k_index]
            ignore_flag = False
            for child_k in children_k:
                if child_k < 0:
                    continue
                if self.is_removed(child_k):
                    ignore_flag = True
            if ignore_flag:
                continue
            fa = self.param.extract(self, k, children_k, children_k_index)
            #global_param_version = self.param.fm.get_param_g().get_version()
            score = fa.get_score(self.param)
            #print('score after get_score:', score)
            # print("hyper edge child: ", children_k, " curr node arr: ", self.get_node_array(k))
            for child_k in children_k:
                if child_k < 0:
                    continue
                score += self.inside_scores[child_k]

            scores[children_k_index] = score

        # print(" curr node arr: ", self.get_node_array(k))
        # print('scores:', scores, " len : ", len(scores))
        # for score in scores:
        #     print('score type:', type(score))
        #print(len(scores))
        self.inside_scores[k] = scores[0]  if len(scores) == 1 else log_sum_exp(scores)


    def touch(self):
        for k in range(self.count_nodes()):
            self.touch_node(k)


    def touch_node(self, k):
        '''
        :param k:
        :return:
        '''
        if self.is_removed(k):
            return

        children_lisk_k = self.get_children(k)
        for children_k_index in range(len(children_lisk_k)):
            children_k = children_lisk_k[children_k_index]
            self.param.extract(self, k, children_k, children_k_index)




    def get_node_array(self, k):
        node = self.get_node(k)
        return NetworkIDMapper.to_hybrid_node_array(node)


    @abstractmethod
    def get_children(self, k) -> np.ndarray:
        pass


    @abstractmethod
    def get_node(self, k):
        pass


    @abstractmethod
    def count_nodes(self) -> int:
        pass


    @abstractmethod
    def is_removed(self, k):
        pass


    def getMaxSharedArray(self):


        if self.maxSharedArray == None or self.count_nodes() > len(self.maxSharedArray):
            self.maxSharedArray = autograd.Variable(
                torch.Tensor(self.count_nodes()).fill_(0.0))  ##np.zeros(self.count_nodes(), dtype=np.float)
        return self.maxSharedArray


    def getMaxPathSharedArray(self):
        if self.maxPathsSharedArrays == None or self.count_nodes() > len(self.maxPathsSharedArrays):
            self.maxPathsSharedArrays = [None for i in range(self.count_nodes())]
        return self.maxPathsSharedArrays

    def is_sum_node(self, k):
        return False


    def max(self):
        self._max = self.getMaxSharedArray()
        self._max_paths = self.getMaxPathSharedArray()
        for k in range(self.count_nodes()):
            self.maxk(k)
        # print("after max: ", len(self._max))
        # print("after max: ", len(self._max_paths))
        # print("after max count: ", self.count_nodes())

    def get_max_path(self, k):
        # print(len(self._max_paths))
        # print(k)

        return self._max_paths[k]

    def maxk(self, k):
        if self.is_removed(k):
            self._max[k] = float("-inf")
            return

        if self.is_sum_node(k):
            inside = 0.0
            children_list_k = self.get_children(k)

            if len(children_list_k) == 0:
                children_list_k = np.zeros((1, 0))

            scores = []
            for children_k_index in range(len(children_list_k)):
                children_k = children_list_k[children_k_index]
                ignore_flag = False
                for child_k in children_k:
                    if child_k < 0:
                        continue
                    if self.is_removed(child_k):
                        ignore_flag = True

                if ignore_flag:
                    continue

                fa = self.param.extract(self, k, children_k, children_k_index)
                global_param_version = self.param.fm.get_param_g().get_version()
                score = fa.get_score(self.param, global_param_version)
                for child_k in children_k:
                    if child_k < 0:
                        continue

                    score += self._max[child_k]

                    # inside = sumLog(inside, score)
                scores.append(score)

            self._max[k] = log_sum_exp(scores)

        else:

            children_list_k = self.get_children(k)
            self._max[k] = float("-inf")

            for children_k_index in range(len(children_list_k)):
                children_k = children_list_k[children_k_index]
                ignore_flag = False
                for child_k in children_k:
                    if child_k < 0:
                        continue
                    if self.is_removed(child_k):
                        ignore_flag = True
                        break

                if ignore_flag:
                    continue

                fa = self.param.extract(self, k, children_k, children_k_index)
                #global_param_version = self.param.fm.get_param_g().get_version()
                score = fa.get_score(self.param)
                for child_k in children_k:
                    if child_k < 0:
                        score += self._max[child_k]

                # print('maxk:',type(score), '\t', type(self._max[k]))
                if score.data[0] >= self._max[k].data[0]:
                    self._max[k] = score
                    self._max_paths[k] = children_k

















import LocalNetworkParam
import NetworkConfig
import FeatureArray
import FeatureBox
from abc import ABC, abstractmethod

class FeatureManager:

    def __init__(self, param_g):
        self.param_g = param_g

        self.numThreads = NetworkConfig.NUM_THREADS
        self._params_l = LocalNetworkParam[self.numThreads]
        self._cachedEnabled = False
        self._num_networks = None
        self._cache = None


    def enable_cache(self, num_networks):
        self._num_networks = num_networks
        self._cache = [None for i in range(num_networks)]
        self._cachedEnabled = True

    def disable_cache(self):


        self._cache = None
        self._cache_enabled = False


    def is_cache_enabled(self):
        return self._cache_enabled


    def get_param_g(self):
        return self._param_g


    def get_params_l(self):
        return self._params_l


    def extract(self, network, parent_k, children_k, children_k_index):
        should_cache = self.is_cache_enable() and (
                (not NetworkConfig.PARALLEL_FEATURE_EXTRACTION) or NetworkConfig.NUM_THREADS == 1 or (
            not NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY) or self._is_finalized)

        if should_cache:
            if self._cache == None:
                self._cache = [None for i in range(self._num_networks)]

            if self._cache[network.get_network_id()] == None:
                self._cache[network.get_network_id()] = [None for i in range(network.count_nodes())]

            if self._cache[network.get_network_id()][parent_k] == None:
                self._cache[network.get_network_id()][parent_k] = [FeatureArray() for i in range(
                    len(network.get_children(parent_k)))]  # TO DO: FeatureArray

            if self._cache[network.get_network_id()][parent_k][children_k_index] != None:
                return self._cache[network.get_network_id()][parent_k][children_k_index]

        fa = self.extract_hepler(network, parent_k, children_k, children_k_index)

        if should_cache:
            self._cache[network.get_network_id()][parent_k][children_k_index] = fa

        return fa


    @abstractmethod
    def extract_helper(self, network, parent_k, children_k, children_k_index):
        pass


    def create_feature_array(self, network, feature_indices, next_fa):
        if (not network.get_instance().is_labeled()) and network.get_instance().get_instance_id() > 0:
            return FeatureArray(fs=feature_indices, next_fa=next_fa)

        if NetworkConfig.AVOID_DUPLICATE_FEATURES:
            return FeatureArray(fb=FeatureBox.get_feature_box(fs=feature_indices, param=self._params_l), next_fa=next_fa)
        else:
            return FeatureArray(fs=feature_indices, next_fa=next_fa)
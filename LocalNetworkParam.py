from Utils import *
import NetworkConfig
import StringIndex
import FeatureArray

class LocalNetworkParam:

    def __init__(self, fm, num_networks):


        self.num_networks = num_networks
        self.fm = fm
        self.obj = 0.0
        self._is_finalized = False
        self._version = 0
        self._global_mode = False
        self._cache = None
        self._cache_enable = True
        self._fs = None

        if not NetworkConfig.CACHE_FEATURES_DURING_TRAINING:
            self.disable_cache()
        if NetworkConfig.NUM_THREADS == 1:
            self._global_mode = True

        self._string_index = StringIndex()


    def set_global_mode(self):
        self._global_mode = True


    def is_global_mode(self):
        return self._global_mode


    def get_features(self):
        return self._fs


    def get_version(self):
        return self._version


    def get_obj(self):
        return self.obj


    def size(self):
        return len(self._fs)


    def reset(self):
        pass
        ### TODO:


    def disable_cache(self):
        self._cache = None
        self._cache_enable = False


    def enable_cache(self):
        self._cache_enable = True


    def is_cache_enabled(self):
        return self._cache_enable


    def extract(self, network, parent_k, children_k, children_k_index):
        '''
      * Extract features from the specified network at current hyperedge, specified by its parent node
         * index (parent_k) and its children node indices (children_k).<br>
         * The children_k_index represents the index of current hyperedge in the list of hyperedges coming out of
         * the parent node.<br>
         * Note that a node with no outgoing hyperedge will still be considered here with empty children_k
         * @param network
         * @param parent_k (int)
         * @param children_k (int[])
         * @param children_k_index (int)
         * @return
        '''
        should_cache = self.is_cache_enable() and (
                    (not NetworkConfig.PARALLEL_FEATURE_EXTRACTION) or NetworkConfig.NUM_THREADS == 1 or (
                not NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY) or self._is_finalized)

        if should_cache:
            if self._cache == None:
                self._cache = [None for i in range(0, self._num_networks)]

            if self._cache[network.get_network_id()] == None:
                self._cache[network.get_network_id()] = [None for i in range(network.count_nodes())]

            if self._cache[network.get_network_id()][parent_k] == None:
                self._cache[network.get_network_id()][parent_k] = [FeatureArray() for i in range(len(network.get_children(parent_k)))]  # TO DO: FeatureArray

            if self._cache[network.get_network_id()][parent_k][children_k_index] != None:
                return self._cache[network.get_network_id()][parent_k][children_k_index]

        fa = self.fm.extract(network, parent_k, children_k, children_k_index)
        if not self.is_global_mode():
            fa = fa.to_local(self)

        if should_cache:
            self._cache[network.get_network_id()][parent_k][children_k_index] = fa

        return fa


    def is_global_mode(self):
        return self._global_mode;


    def finalize_it(self):
        if (self.is_global_mode()):
            eprint("Finalizing local features in global mode: not required")
        self._is_finalized = True
        return

    ## TODO code for non-global mode
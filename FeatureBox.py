import NetworkConfig

class FeatureBox:

    def __init__(self, fs):
        '''
        fs is a list, not a Tensor
        '''

        self._fs = fs  ## list, not tensor yet

        self._version = -1


    def length(self):


        return len(self._fs)


    def get(self):
        return self._fs


    def get_pos(self, pos):
        return self._fs[pos]

    def get_feature_box(self, fs, param):
        fb = FeatureBox(fs)

        if not NetworkConfig.AVOID_DUPLICATE_FEATURES:
            return fb
        if param.fb_map == None:
            param.fb_map = {}
        if fb in param.fb_map:
            return param.fb_map[fb]
        else:
            param.fb_map[fb] = fb
            return fb


    def __hash__(self):
        return hash(self._fs)


    def __eq__(self, other):
        if isinstance(other, FeatureBox):
            return (self._fs == other._fs)

        return False
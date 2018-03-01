import math
from NetworkConfig import NetworkConfig
from Utils import *
from FeatureBox import FeatureBox


class FeatureArray:

    NEGATIVE_INFINITY = None
    EMPTY = None

    def __init__(self, fs=None, fb=None, next_fa=None, total_score=None):

        # if FeatureArray.NEGATIVE_INFINITY is None:
        #     # Create and remember instance
        #     FeatureArray.NEGATIVE_INFINITY = FeatureArray.__impl()()
        self._fb = None
        self._fs = None
        self._next = None
        self._total_score = None

        if fs == None and fb == None and total_score == None:
            raise Exception("fs and fb can't be both None (when total score is none)")
        if fs != None and fb != None:
            raise Exception("fs and fb couldn't initialize at the same time")
        if fs != None:
            self._fb = FeatureBox(fs=fs)
        elif fb != None:
            self._fb = fb
        else:
            self._total_score = total_score
        self._next = next_fa
        self._is_local = False



    def to_local(self, param):
        if self == FeatureArray.NEGATIVE_INFINITY:
            return self
        if self._is_local:
            return self
        length = self._fb.length()
        if NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY:
            for fs in self._fb.get():
                if fs == -1:
                    length = length - 1

        fs_local = [None] * length
        local_idx = 0

        for k in range(self._fb.length()):
            if self._fb.get(k) == -1 and NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY:
                local_idx = local_idx - 1
                continue
            if not NetworkConfig.PARALLEL_FEATURE_EXTRACTION or NetworkConfig.NUM_THREADS == 1 or param._is_finalized:
                pass
                ###TODO: there is no tolocal in param
            else:
                fs_local[local_idx] = self._fb.get(k)
            if fs_local[local_idx] == -1:
                raise Exception("The local feature got an id of -1 for ", self._fb.get(k))
            local_idx = local_idx + 1

        fa = None
        if self._next != None:
            fa = FeatureArray(fb=FeatureBox.get_feature_box(fs_local, param), next_fa = self._next.to_local(param))
        else:
            fa = FeatureArray(fb=FeatureBox.get_feature_box(fs_local, param))

        fa._is_local = True

        return fa


    def get_current(self):
        return self._fb.get()


    def next(self, next_fa):
        self._next = next_fa


    def get_next(self):
        return self._next


    def add_next(self, next_fa):
        self._next = next_fa
        return self._next


    def get_score(self, param, version = None):
        if (self == FeatureArray.NEGATIVE_INFINITY):
            return self._total_score

        if (not self._is_local) != param.is_global_mode():
            eprint(self._next)
            raise Exception('This FeatureArray is local? ', self._is_local, '; The param is ', param.is_global_mode())

        # print('self._total_score:', self._total_score, '\t', type(self._total_score))
        # if self._total_score == -math.inf:
        #     return self._total_score

        self._total_score = 0.0
        #if self._fb._version != version:
        self._fb._curr_score = self.compute_score(param, self.get_current())
        self._fb._version = version

        self._total_score += self._fb._curr_score

        if self._next != None:
            self._total_score += self._next.get_score(param, version)

        return self._total_score


    def compute_score(self, param, fs, fvs=None):
        if (not self._is_local) != param.is_global_mode():
            raise Exception('This FeatureArray is local? ', self._is_local, '; The param is ', param.is_global_mode())

        score = autograd.Variable(torch.Tensor(1).fill_(0.0))
        for i in range(len(fs)):
            f = fs[i]
            #fv = 1 if fvs != None else fvs[i]  ## TODO: use m.index_select(dim, torch.LongTensor(list))
            if f != -1:
                score += param.get_weight(f)

        return score


    def size(self):
        size = self._fb.length()
        if (self._next != None):
            size += self._next.size()

        return size


    def __hash__(self):

        code = hash(self._fb)
        if self._fb != None:
            code = code ^ hash(self._next)

        return code


    def __eq__(self, other):
        if isinstance(other, FeatureArray):

            #empty equality
            if self._fs != None and self._fs == []:
                return other._fs != None and other._fs == []

            #negative infinity equality
            if self._fb == None:
                return other._fb == None

            if not (self._fb == other._fb):
                return False

            if self._next == None:
                if other._next != None:
                    return False
                else:
                    return True
            else:
                return self._next.equals(other._next)

        return False




FeatureArray.NEGATIVE_INFINITY = FeatureArray(total_score=-math.inf)
FeatureArray.EMPTY = FeatureArray(fs=[])



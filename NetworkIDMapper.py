from __future__ import print_function
import sys
import numpy as np
from NetworkConfig import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class NetworkIDMapper:
    CAPACITY = NetworkConfig.DEFAULT_CAPACITY_NETWORK

    @staticmethod
    def get_capacity():
        return NetworkIDMapper.CAPACITY

    # _CAPACITY_NETWORK = capacity;
    # //now we check if the capacity is valid.
    # int[] v = new int[capacity.length];
    # for(int k = 0; k<v.length; k++)
    # v[k] = capacity[k]-1;
    # int[] u = NetworkIDMapper.toHybridNodeArray(toHybridNodeID(v));
    # if(!Arrays.equals(u, v)){
    # throw new RuntimeException("The capacity appears to be too large:"+Arrays.toString(capacity));
    # }
    # System.err.println("Capacity successfully set to: "+Arrays.toString(capacity));
    @staticmethod
    def set_capacity(new_capacity):
        v = np.zeros(len(NetworkIDMapper.CAPACITY))
        for k in range(len(v)):
            v[k] = new_capacity[k] - 1
            u = NetworkIDMapper.to_hybrid_node_array(NetworkIDMapper.to_hybrid_node_ID(v))
            print(u,v)
            if not np.array_equal(u, v):
                raise Exception("The capacity appears to be too large: ", new_capacity)
        eprint("Capacity successfully set to: ", new_capacity)

    @staticmethod
    def to_hybrid_node_array(value):
        result = np.zeros(len(NetworkIDMapper.CAPACITY))

        for k in range(len(result) - 1, 0, -1):
            v = value // NetworkIDMapper.CAPACITY[k]
            print(value, v)
            result[k] = value % NetworkIDMapper.CAPACITY[k]
            value = v
        return result


    @staticmethod
    def to_hybrid_node_ID(array):
        if len(array) != len(NetworkIDMapper.CAPACITY):
            raise Exception("array size is ", len(array))



        #v = np.int64(0)
        #print(type(v), type(array[0]))
        v = array[0]

        for k in range(1, len(array)):
            if array[k] >= NetworkIDMapper.CAPACITY[k]:
                raise Exception("Invalid: capacity for ", k, " is ", NetworkIDMapper.CAPACITY[k], " but the value is ", array[k])

            #print(type(array[k]), type(NetworkIDMapper.CAPACITY[k]), type(v))
            v = v * NetworkIDMapper.CAPACITY[k] + array[k]
            #print(NetworkIDMapper.CAPACITY[k], array[k], v, type(v))

        return v


if __name__ == "__main__":
    print('to_hybrid_node_ID:')
    #NetworkIDMapper.set_capacity(np.asarray([1000, 1000, 1000]))
    a = NetworkIDMapper.to_hybrid_node_ID(np.asarray([100, 1]))
    print(a)
    b = NetworkIDMapper.to_hybrid_node_array(a)
    print(b)
from __future__ import print_function
import sys
import numpy as np
import NetworkConfig


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class NetworkIDMapper:
    CAPACITY = NetworkConfig.DEFAULT_CAPACITY_NETWORK

    @staticmethod
    def get_capacity():
        return CAPACITY

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
        v = np.zeros(len(CAPACITY))
        for k in range(len(v)):
            v[k] = new_capacity[k] - 1
            u = to_hybrid_node_array(to_hybrid_node_ID(v))
            if not np.array_equal(u, v):
                raise Exception("The capacity appears to be too large: ", new_capacity)
        eprint("Capacity successfully set to: ", new_capacity)

        @staticmethod


    def to_hybrid_node_array(value):
        result = np.zeros(len(CAPACITY))


    for k in range(_RESULT.length - 1, 0, -1):
        v = value / CAPACITY[k]
        result[k] = value % CAPACITY[k]
        value = v
    return result


    @staticmethod
    def to_hybrid_node_ID(array):
        if len(array) != len(CAPACITY):
            raise Exception("array size is ", len(array))

        v = array[0]
        for k in range(len(array)):
            if array[k] >= CAPACITY[k]:
                raise Exception("Invalid: capacity for ", k, " is ", CAPACITY[k], " but the value is ", array[k])
            v = v * CAPACITY[k] = array[k]

        return v
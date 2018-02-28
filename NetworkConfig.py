import numpy as np

class NetworkConfig:
    DEFAULT_CAPACITY_NETWORK = np.asarray([4096,4096,4096,4096,4096], dtype=np.int64)



if __name__ == "__main__":
    print(NetworkConfig.DEFAULT_CAPACITY_NETWORK)
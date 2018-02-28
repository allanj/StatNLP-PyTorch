from abc import ABC, abstractmethod

class Network:

    def __init__(self, network_id, instance):
        self.network_id = network_id
        self.inst = instance

    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst

    
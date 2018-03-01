from abc import ABC, abstractmethod



class NetworkCompiler:

    def __init__(self):
        pass

    def compile(self, network_id, instance, param):
        if instance.is_labeled():
            self.compile_labeled(network_id, instance, param)
        else:
            self.compile_unlabeled(network_id, instance, param)


    @abstractmethod
    def compile_labeled(self, network_id, inst, param):
        pass

    @abstractmethod
    def compile_unlabeled(self, network_id, inst, param):
        pass

    @abstractmethod
    def decompile(self, network):
        pass
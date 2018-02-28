class StringIndex:

    def __init__(self):
        self.index = {}
        self.locked = False
        self.array = None

    def get_or_put(self, str):
        if str in self.index:
            if self.locked:
                return -1
            self.index[str] = len(self.index)
        return self.index[str]

    def put(self, str):
        return self.get_or_put(str)

    def get_id(self, str):
        return self.index[str]

    def lock(self):
        self.locked = True

    def build_reverse_index(self):
        self.array = ['' for i in range(len(self.index))]
        for name in self.index:
            id = self.index[name]
            self.array[id] = name

    def remove_reverse_index(self):
        self.array = None

    def get_str(self, id):
        return self.array[id]

    def size(self):
        return len(self.index)

    def keys(self):
        return self.index.keys()

    @staticmethod
    def merge(self, indexes):
        pass


pass
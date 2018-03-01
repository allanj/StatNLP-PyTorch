

class Test():


    na = None

    def __init__(self, k, static = False):
        self.k = k

        # if static == True:
        #     pass
        # else:
        #     if Test.na == None:
        #         na = Test(100, static=True)


Test.na = Test(100)


if __name__ == "__main__":
    a = Test(99)
    print(a.k)
    print(Test.na.k)
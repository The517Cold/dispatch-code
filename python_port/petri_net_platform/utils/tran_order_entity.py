class TranOrderEntity:
    def __init__(self, tran, times):
        self.tran = tran
        self.times = times

    def get_tran(self):
        return self.tran

    def __eq__(self, other):
        if not isinstance(other, TranOrderEntity):
            return False
        return self.tran == other.tran and self.times == other.times

    def __hash__(self):
        return hash((self.tran, self.times))

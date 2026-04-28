class TranLink:
    def __init__(self, curr_tran, pre):
        self.curr_tran = curr_tran
        self.pre = pre


class LowSpaceLink:
    def __init__(self, tran_link, curr):
        self.tran_link = tran_link
        self.curr = curr

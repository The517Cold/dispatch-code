class AbstractSearch:
    def __init__(self):
        self.end = None
        self.petri_net = None

    def same(self, curr):
        p_info = curr.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            if p_info[i] != self.end[i]:
                return False
        return True

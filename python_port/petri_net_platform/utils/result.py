class Result:
    def __init__(self, trans, markings):
        self.trans = trans
        self.markings = markings

    def get_trans(self):
        return self.trans

    def get_markings(self):
        return self.markings

    def __str__(self):
        trans_buffer = []
        marking_buffer = []
        trans_buffer.append("变迁序列:")
        if self.trans:
            trans_buffer.append("->".join(str(v) for v in self.trans))
        marking_buffer.append("标识序列:")
        for i in range(len(self.markings) - 1):
            marking_buffer.append("标识" + str(i) + ": " + str(self.markings[i]))
        if self.markings:
            last = len(self.markings) - 1
            marking_buffer.append("标识" + str(last) + ": " + str(self.markings[last]))
        return "\n".join(trans_buffer) + "\n" + "\n".join(marking_buffer)

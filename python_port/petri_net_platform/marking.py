from collections import deque
from .architecture import Marking, HasResideceTime


class AbstractMarking(Marking):
    def __init__(self, prefix=0):
        self.prefix = prefix

    def __hash__(self):
        return hash(tuple(self.get_p_info()))

    def __eq__(self, other):
        if not isinstance(other, Marking):
            return False
        return list(self.get_p_info()) == list(other.get_p_info())

    def __str__(self):
        p_info = self.get_p_info()
        if not p_info:
            return "() 全局时间: " + str(self.get_prefix())
        items = ",".join(str(v) for v in p_info)
        return "(" + items + ") 全局时间: " + str(self.get_prefix())

    def get_prefix(self):
        return self.prefix

    def clone(self):
        raise NotImplementedError()


class AbstractMarkingWithDeterminedPInfo(AbstractMarking):
    def __init__(self, p_info, prefix=0):
        super().__init__(prefix)
        self.p_info = p_info

    def get_p_info(self):
        return self.p_info


class NormalMarking(AbstractMarkingWithDeterminedPInfo):
    def __init__(self, p_info, prefix):
        super().__init__(p_info, prefix)

    def clone(self):
        return NormalMarking(self.p_info.copy(), self.prefix)


class TTimeMarking(AbstractMarkingWithDeterminedPInfo):
    def __init__(self, p_info, prefix, curr_delay_t):
        super().__init__(p_info, prefix)
        self.curr_delay_t = curr_delay_t
        self.is_enable = [False] * len(curr_delay_t)
        self.nexts = {}

    def clone(self):
        clone = TTimeMarking(self.p_info.copy(), self.prefix, self.curr_delay_t.copy())
        clone.is_enable = self.is_enable.copy()
        clone.nexts = {tran: p_info.copy() for tran, p_info in self.nexts.items()}
        return clone


class TPPNMarking(AbstractMarkingWithDeterminedPInfo):
    def __init__(self, p_info, t_info, prefix):
        super().__init__(p_info, prefix)
        self.t_info = t_info

    def clone(self):
        t_info_copy = [list(v) for v in self.t_info]
        return TPPNMarking(self.p_info.copy(), t_info_copy, self.prefix)


class TTPPNMarking(AbstractMarkingWithDeterminedPInfo):
    def __init__(self, p_info, t_info, prefix):
        super().__init__(p_info, prefix)
        self.t_info = t_info
        self.curr_delay_t = []
        self.is_enable = []
        self.nexts = {}

    def remove_tokens(self, dq, count):
        min_val = 0
        for _ in range(count):
            min_val = dq.popleft()
        return min_val

    def get_the_kth_small(self, dq, k):
        idx = 0
        for val in dq:
            idx += 1
            if idx >= k:
                return val
        return -1

    def clone(self):
        t_info_copy = [deque(list(v)) for v in self.t_info]
        clone = TTPPNMarking(self.p_info.copy(), t_info_copy, self.prefix)
        clone.curr_delay_t = self.curr_delay_t.copy()
        clone.is_enable = self.is_enable.copy()
        clone.nexts = {tran: p_info.copy() for tran, p_info in self.nexts.items()}
        return clone


class TTPPNMarkingHasResidenceTime(TTPPNMarking, HasResideceTime):
    def __init__(self, p_info, t_info, prefix, residence_time_info):
        super().__init__(p_info, t_info, prefix)
        self.residence_time_info = residence_time_info
        self.over_max_residence_time = False
        self.tran_last_enable_time = 0
        self.last_enable_times = []
        self.over_residence_time_place = 0

    def clone(self):
        p_info_copy = self.p_info.copy()
        t_info_copy = [deque(list(v)) for v in self.t_info]
        residence_copy = [deque(list(v)) for v in self.residence_time_info]
        clone = TTPPNMarkingHasResidenceTime(p_info_copy, t_info_copy, self.prefix, residence_copy)
        clone.over_max_residence_time = self.over_max_residence_time
        clone.over_residence_time_place = self.over_residence_time_place
        clone.tran_last_enable_time = self.tran_last_enable_time
        clone.last_enable_times = self.last_enable_times.copy()
        clone.curr_delay_t = self.curr_delay_t.copy()
        clone.is_enable = self.is_enable.copy()
        clone.nexts = {tran: p_info.copy() for tran, p_info in self.nexts.items()}
        return clone

    def __eq__(self, other):
        if not isinstance(other, TTPPNMarkingHasResidenceTime):
            return False
        return list(self.p_info) == list(other.p_info) and self.over_max_residence_time == other.over_max_residence_time

    def __hash__(self):
        return hash((tuple(self.p_info), self.over_max_residence_time))

    def __str__(self):
        items = ",".join(str(v) for v in self.p_info)
        base = "(" + items + ") 全局时间: " + str(self.prefix) + "    变迁使能时间:" + str(self.tran_last_enable_time)
        parts = [base, "驻留时间:"]
        for i in range(len(self.residence_time_info) - 1):
            parts.append("库所" + str(i) + "驻留时间: " + str(list(self.residence_time_info[i])))
        last = len(self.residence_time_info) - 1
        parts.append("库所" + str(last) + "驻留时间: " + str(list(self.residence_time_info[last])))
        return "\n".join(parts)

    def is_over_residece_time(self):
        return self.over_max_residence_time

    def get_residence_time(self, place, token=None):
        if token is None:
            return sum(self.residence_time_info[place])
        idx = 0
        for time in self.residence_time_info[place]:
            if idx >= token:
                return time
            idx += 1
        return -1

    def get_over_residence_time_place(self):
        return self.over_residence_time_place

    def get_time(self, place):
        return sum(self.t_info[place])


class Token:
    def __init__(self, token_id, timer, residence_time):
        self.id = token_id
        self.timer = timer
        self.residence_time = residence_time

    def get_id(self):
        return self.id

    def clone(self):
        return Token(self.id, self.timer, self.residence_time)


class TTPPNMarkingByTokenWithResTime(HasResideceTime):
    def __init__(self, p_info, min_delay_t, min_delay_p):
        self.max_id = 0
        self.prefix = 0
        self.t_info = []
        self.curr_delay_t = min_delay_t.copy()
        self.is_enable = []
        self.qtime_map = {}
        self.over_max_residence_time = False
        self.over_residence_time_place = 0
        self.consume = 0
        self.waste = 0
        for place, count in enumerate(p_info):
            dq = deque()
            for _ in range(count):
                dq.append(Token(self.max_id, min_delay_p[place], 0))
                self.max_id += 1
            self.t_info.append(dq)

    def remove_tokens(self, dq, count):
        tokens = []
        for _ in range(count):
            tokens.append(dq.popleft())
        return tokens

    def get_the_kth_small(self, dq, k):
        idx = 0
        for token in dq:
            idx += 1
            if idx >= k:
                return token.timer
        return -1

    def get_prefix(self):
        return self.prefix

    def get_p_info(self):
        return [len(dq) for dq in self.t_info]

    def clone(self):
        clone = TTPPNMarkingByTokenWithResTime.__new__(TTPPNMarkingByTokenWithResTime)
        clone.t_info = [deque(Token(t.get_id(), t.timer, t.residence_time) for t in dq) for dq in self.t_info]
        clone.max_id = self.max_id
        clone.prefix = self.prefix
        clone.curr_delay_t = self.curr_delay_t.copy()
        clone.is_enable = self.is_enable.copy()
        clone.qtime_map = dict(self.qtime_map)
        clone.over_max_residence_time = self.over_max_residence_time
        clone.over_residence_time_place = self.over_residence_time_place
        clone.consume = self.consume
        clone.waste = self.waste
        return clone

    def __eq__(self, other):
        if not isinstance(other, TTPPNMarkingByTokenWithResTime):
            return False
        return self.over_max_residence_time == other.over_max_residence_time and self.get_p_info() == other.get_p_info()

    def __hash__(self):
        return hash((tuple(self.get_p_info()), self.over_max_residence_time))

    def __str__(self):
        p_info = self.get_p_info()
        items = ",".join(str(v) for v in p_info)
        lines = ["(" + items + ") 全局时间:" + str(self.prefix)]
        for place in range(len(self.t_info) - 1):
            token_desc = " ".join(str(t.get_id()) + "[" + str(t.residence_time) + "]" for t in self.t_info[place])
            lines.append("库所" + str(place) + " token情况:" + token_desc + "  ")
        last = len(self.t_info) - 1
        token_desc = " ".join(str(t.get_id()) for t in self.t_info[last])
        lines.append("库所" + str(last) + " token情况:" + token_desc + " ")
        return "\n".join(lines)

    def is_over_residece_time(self):
        return self.over_max_residence_time

    def get_residence_time(self, place, token=None):
        if token is None:
            return sum(t.residence_time for t in self.t_info[place])
        idx = 0
        for t in self.t_info[place]:
            if idx >= token:
                return t.residence_time
            idx += 1
        return -1

    def get_over_residence_time_place(self):
        return self.over_residence_time_place

    def get_time(self, place):
        return sum(t.timer for t in self.t_info[place])

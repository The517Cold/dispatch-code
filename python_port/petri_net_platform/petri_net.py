from collections import deque
from .architecture import PetriNet
from .marking import NormalMarking, TPPNMarking, TTPPNMarking, TTPPNMarkingHasResidenceTime, TTPPNMarkingByTokenWithResTime, TTimeMarking, Token


class NormalPetriNet(PetriNet):
    def __init__(self, p_info, a_matrix):
        self.curr = NormalMarking(p_info, 0)
        self.a_matrix = a_matrix

    def launch(self, tran):
        next_marking = self.curr.clone()
        for i in range(len(self.a_matrix)):
            next_marking.p_info[i] += self.a_matrix[i][tran]
        next_marking.prefix += 1
        return next_marking

    def enable(self, tran):
        p_info = self.curr.p_info.copy()
        count_p = len(self.a_matrix)
        for i in range(count_p):
            p_info[i] += self.a_matrix[i][tran]
            if p_info[i] < 0:
                return False
        return True

    def get_marking(self):
        return self.curr

    def set_marking(self, marking):
        self.curr = marking

    def get_trans_count(self):
        return len(self.a_matrix[0])

    def clone(self):
        return NormalPetriNet(self.curr.p_info.copy(), self.a_matrix)


class NormalPetriNetHasPostAndPre(NormalPetriNet):
    def __init__(self, p_info, a_matrix, pre, capacity=None):
        self.pre = pre
        self.capacity = capacity
        self.has_capacity = capacity is not None
        self.curr = NormalMarking(p_info, 0)
        self.a_matrix = a_matrix

    def enable(self, tran):
        p_info = self.curr.p_info.copy()
        count_p = len(self.a_matrix)
        for i in range(count_p):
            p_temp = p_info[i] - self.pre[i][tran]
            p_info[i] += self.a_matrix[i][tran]
            if p_temp < 0 or (self.has_capacity and p_info[i] > self.capacity[i]):
                return False
        return True


class TPPN(PetriNet):
    def __init__(self, p_info, a_matrix, min_delay_p):
        t_info = []
        for i in range(len(p_info)):
            lst = []
            for _ in range(p_info[i]):
                lst.append(min_delay_p[i])
            t_info.append(lst)
        self.curr = TPPNMarking(p_info, t_info, 0)
        self.a_matrix = a_matrix
        self.min_delay_p = min_delay_p

    def launch(self, tran):
        next_marking = self.curr.clone()
        count_p = len(self.a_matrix)
        max_time = 0
        add_list = []
        for i in range(count_p):
            s_t_inf = next_marking.t_info[i]
            if self.a_matrix[i][tran] < 0:
                k = -self.a_matrix[i][tran]
                min_list = self._k_min(s_t_inf, k)
                min_val = min_list[0][0] if min_list else 0
                self._mark_copy(min_list, s_t_inf)
                max_time = max(max_time, min_val)
                next_marking.p_info[i] -= k
            else:
                k = self.a_matrix[i][tran]
                for _ in range(k):
                    add_list.append(i)
                next_marking.p_info[i] += k
        for lst in next_marking.t_info:
            for idx in range(len(lst)):
                val = lst[idx] - max_time
                lst[idx] = val if val > 0 else 0
        for i in add_list:
            next_marking.t_info[i].append(self.min_delay_p[i])
        next_marking.prefix += max_time
        return next_marking

    def enable(self, tran):
        p_info = self.curr.p_info.copy()
        for i in range(len(self.a_matrix)):
            p_info[i] += self.a_matrix[i][tran]
            if p_info[i] < 0:
                return False
        return True

    def _mark_copy(self, min_list, s_t_inf):
        for _, idx in min_list:
            s_t_inf[idx] = -1
        i = 0
        for j in range(len(s_t_inf)):
            value = s_t_inf[j]
            if value != -1:
                s_t_inf[i] = value
                i += 1
        while len(s_t_inf) > i:
            s_t_inf.pop(i)

    def _k_min(self, s_t_inf, k):
        min_list = []
        for count in range(min(k, len(s_t_inf))):
            min_list.append([s_t_inf[count], count])
        min_list.sort(reverse=True)
        for count in range(k, len(s_t_inf)):
            if min_list[0][0] > s_t_inf[count]:
                min_list[0] = [s_t_inf[count], count]
                min_list.sort(reverse=True)
        return min_list

    def get_marking(self):
        return self.curr

    def get_trans_count(self):
        return len(self.a_matrix[0])

    def clone(self):
        return TPPN(self.curr.p_info.copy(), self.a_matrix, self.min_delay_p)

    def set_marking(self, marking):
        self.curr = marking


class TTPPN(PetriNet):
    def __init__(self, p_info, a_matrix, min_delay_p, min_delay_t):
        t_info = []
        for i in range(len(p_info)):
            dq = deque()
            for _ in range(p_info[i]):
                dq.append(min_delay_p[i])
            t_info.append(dq)
        self.curr = TTPPNMarking(p_info, t_info, 0)
        self.curr.curr_delay_t = min_delay_t.copy()
        self.min_delay_p = min_delay_p
        self.a_matrix = a_matrix
        self.min_delay_t = min_delay_t
        self.t_list = [[] for _ in range(len(a_matrix))]
        self.p_list = [[] for _ in range(len(a_matrix[0]))]
        for place in range(len(a_matrix)):
            for tran in range(len(a_matrix[0])):
                if a_matrix[place][tran] < 0:
                    self.t_list[place].append(tran)
        for tran in range(len(a_matrix[0])):
            for place in range(len(a_matrix)):
                if a_matrix[place][tran] < 0:
                    self.p_list[tran].append(place)
        self.t_launch_is_enable = []
        self.before_tlaunch_timer = []
        self.time = 0
        self.need_remove = set()
        self._set_next(self.curr)

    def launch(self, tran):
        self.before_tlaunch_timer = [0] * len(self.a_matrix[0])
        add_list = []
        self._before_tlaunch(tran, add_list)
        self._tlaunch(tran)
        self._after_tlaunch(self.time, add_list)
        return self.next

    def _before_tlaunch(self, tran, add_list):
        self.next = self.curr.clone()
        next_p_info = self.curr.nexts.get(tran)
        self.next.p_info = next_p_info
        self.t_launch_is_enable = self.curr.is_enable.copy()
        self.need_remove = set()
        curr_p_info = self.curr.p_info
        max_time = 0
        for i in range(len(curr_p_info)):
            add = next_p_info[i] - curr_p_info[i]
            s_t_inf = self.next.t_info[i]
            if add < 0:
                self.need_remove.add(i)
                min_val = self.next.remove_tokens(s_t_inf, -add)
                max_time = max(max_time, min_val)
            else:
                for _ in range(add):
                    add_list.append(i)
        self.time = max_time

    def _tlaunch(self, tran):
        places = self.p_list[tran]
        next_delay_t = self.curr.curr_delay_t.copy()
        delay_time = next_delay_t[tran]
        self.time += delay_time
        for tran_next in range(len(self.a_matrix[0])):
            if self.curr.is_enable[tran_next] and self._check_is_enable(tran_next):
                next_delay_t[tran_next] -= self.before_tlaunch_timer[tran_next]
                if next_delay_t[tran_next] < 0:
                    next_delay_t[tran_next] = 0
        for place in places:
            for tran_next in self.t_list[place]:
                next_delay_t[tran_next] = self.min_delay_t[tran_next]
        self.next.curr_delay_t = next_delay_t
        self._set_next(self.next)
        self.next.curr_delay_t[tran] = self.min_delay_t[tran]
        self.next.prefix += self.time

    def _after_tlaunch(self, time, add_list):
        for place in range(len(self.next.t_info)):
            if self.min_delay_p[place] == 0:
                continue
            tokens = self.next.t_info[place]
            size = len(tokens)
            for _ in range(size):
                token_time = tokens.pop() - time
                token_time = max(0, token_time)
                tokens.appendleft(token_time)
        for i in add_list:
            self.next.t_info[i].append(self.min_delay_p[i])

    def _check_is_enable(self, tran):
        next_p_info = self.curr.nexts.get(tran)
        curr_p_info = self.curr.p_info
        t_info = self.curr.t_info
        min_val = 2 ** 31 - 1
        places = self.p_list[tran]
        for place in places:
            if place in self.need_remove:
                return False
            s_t_inf = t_info[place]
            get_val = curr_p_info[place] - next_p_info[place]
            max_val = self.next.get_the_kth_small(s_t_inf, get_val)
            if max_val - self.time > 0:
                return False
            min_val = min(min_val, self.time - max_val)
        self.before_tlaunch_timer[tran] = min_val
        return True

    def _set_next(self, next_marking):
        next_marking.is_enable = [False] * len(self.a_matrix[0])
        next_marking.nexts = {}
        for tran in range(len(self.a_matrix[0])):
            next_p_info = next_marking.p_info.copy()
            is_enable = True
            for place in range(len(self.a_matrix)):
                next_p_info[place] += self.a_matrix[place][tran]
                if next_p_info[place] < 0:
                    is_enable = False
                    break
            if is_enable:
                next_marking.is_enable[tran] = True
                next_marking.nexts[tran] = next_p_info

    def enable(self, tran):
        return self.curr.is_enable[tran]

    def get_marking(self):
        return self.curr

    def set_marking(self, marking):
        self.curr = marking

    def get_trans_count(self):
        return len(self.a_matrix[0])

    def clone(self):
        return TTPPN(self.curr.p_info.copy(), self.a_matrix, self.min_delay_p, self.min_delay_t)


class TTPPNHasPostAndPre(TTPPN):
    def __init__(self, p_info, a_matrix, pre, min_delay_p, min_delay_t, capacity=None):
        self.pre = pre
        self.capacity = capacity
        self.has_capacity = capacity is not None
        t_info = []
        for i in range(len(p_info)):
            dq = deque()
            for _ in range(p_info[i]):
                dq.append(min_delay_p[i])
            t_info.append(dq)
        self.curr = TTPPNMarking(p_info, t_info, 0)
        self.curr.curr_delay_t = min_delay_t.copy()
        self.min_delay_p = min_delay_p
        self.a_matrix = a_matrix
        self.min_delay_t = min_delay_t
        self.t_list = [[] for _ in range(len(a_matrix))]
        self.p_list = [[] for _ in range(len(a_matrix[0]))]
        for place in range(len(a_matrix)):
            for tran in range(len(a_matrix[0])):
                if pre[place][tran] > 0:
                    self.t_list[place].append(tran)
        for tran in range(len(a_matrix[0])):
            for place in range(len(a_matrix)):
                if pre[place][tran] > 0:
                    self.p_list[tran].append(place)
        self.t_launch_is_enable = []
        self.before_tlaunch_timer = []
        self.time = 0
        self.need_remove = set()
        self._set_next(self.curr)

    def _before_tlaunch(self, tran, add_list):
        self.next = self.curr.clone()
        next_p_info = self.curr.nexts.get(tran)
        self.next.p_info = next_p_info
        self.t_launch_is_enable = self.curr.is_enable.copy()
        self.need_remove = set()
        curr_p_info = self.curr.p_info
        max_time = 0
        for i in range(len(curr_p_info)):
            get_val = self.pre[i][tran]
            add = next_p_info[i] - curr_p_info[i] + get_val
            s_t_inf = self.next.t_info[i]
            if get_val > 0:
                self.need_remove.add(i)
                min_val = self.next.remove_tokens(s_t_inf, get_val)
                max_time = max(max_time, min_val)
            if add > 0:
                for _ in range(add):
                    add_list.append(i)
        self.time = max_time

    def _check_is_enable(self, tran):
        min_val = 2 ** 31 - 1
        places = self.p_list[tran]
        t_info = self.curr.t_info
        for place in places:
            if place in self.need_remove:
                return False
            s_t_inf = t_info[place]
            get_val = self.pre[place][tran]
            max_val = self.next.get_the_kth_small(s_t_inf, get_val)
            if max_val - self.time > 0:
                return False
            min_val = min(min_val, self.time - max_val)
        self.before_tlaunch_timer[tran] = min_val
        return True

    def _set_next(self, next_marking):
        next_marking.is_enable = [False] * len(self.a_matrix[0])
        next_marking.nexts = {}
        for tran in range(len(self.a_matrix[0])):
            next_p_info = next_marking.p_info.copy()
            is_enable = True
            for place in range(len(self.a_matrix)):
                value = next_p_info[place] - self.pre[place][tran]
                next_p_info[place] += self.a_matrix[place][tran]
                if value < 0:
                    is_enable = False
                    break
                if self.has_capacity and self.capacity[place] < next_p_info[place]:
                    is_enable = False
                    break
            if is_enable:
                next_marking.is_enable[tran] = True
                next_marking.nexts[tran] = next_p_info

    def clone(self):
        if self.has_capacity:
            return TTPPNHasPostAndPre(self.curr.p_info.copy(), self.a_matrix, self.pre, self.min_delay_p, self.min_delay_t, self.capacity)
        return TTPPNHasPostAndPre(self.curr.p_info.copy(), self.a_matrix, self.pre, self.min_delay_p, self.min_delay_t)


class TTPPNHasResidenceTime(PetriNet):
    def __init__(self, p_info, a_matrix, pre, min_delay_p, min_delay_t, max_residence_time, capacity=None, place_from_places=None, is_resource=None):
        self.pre = pre
        self.capacity = capacity
        self.has_capacity = capacity is not None
        t_info = []
        residence_time_info = []
        for i in range(len(p_info)):
            dq = deque()
            rdq = deque()
            for _ in range(p_info[i]):
                dq.append(min_delay_p[i])
                rdq.append(0)
            t_info.append(dq)
            residence_time_info.append(rdq)
        self.curr = TTPPNMarkingHasResidenceTime(p_info, t_info, 0, residence_time_info)
        self.curr.curr_delay_t = min_delay_t.copy()
        self.min_delay_p = min_delay_p
        self.a_matrix = a_matrix
        self.min_delay_t = min_delay_t
        self.max_residence_time = max_residence_time
        if place_from_places is None:
            place_from_places = [[] for _ in range(len(a_matrix))]
        if is_resource is None:
            is_resource = [False] * len(a_matrix)
        if len(place_from_places) != len(a_matrix):
            place_from_places = [[] for _ in range(len(a_matrix))]
        if len(is_resource) != len(a_matrix):
            is_resource = [False] * len(a_matrix)
        self.place_from_places = place_from_places
        self.is_resource = is_resource
        self.t_list = [[] for _ in range(len(a_matrix))]
        self.p_list = [[] for _ in range(len(a_matrix[0]))]
        self.transition_flow_allowed = [True] * len(a_matrix[0])
        self.curr.last_enable_times = [-1] * len(a_matrix[0])
        for place in range(len(a_matrix)):
            for tran in range(len(a_matrix[0])):
                if pre[place][tran] > 0:
                    self.t_list[place].append(tran)
        for tran in range(len(a_matrix[0])):
            for place in range(len(a_matrix)):
                if pre[place][tran] > 0:
                    self.p_list[tran].append(place)
        self.t_launch_is_enable = []
        self.before_tlaunch_timer = []
        self.time = 0
        self.need_remove = set()
        self._init_transition_flow_constraints()
        self._set_next(self.curr)

    def _init_transition_flow_constraints(self):
        trans_count = len(self.a_matrix[0])
        place_count = len(self.a_matrix)
        for tran in range(trans_count):
            ok = True
            for to_place in range(place_count):
                from_places = self.place_from_places[to_place]
                if not from_places:
                    continue
                produced = self.a_matrix[to_place][tran] + self.pre[to_place][tran]
                if produced <= 0:
                    continue
                available = 0
                for from_place in from_places:
                    if 0 <= from_place < place_count:
                        available += self.pre[from_place][tran]
                if available < produced:
                    ok = False
                    break
            self.transition_flow_allowed[tran] = ok

    def launch(self, tran):
        self.before_tlaunch_timer = [0] * len(self.a_matrix[0])
        add_list = []
        self._before_tlaunch(tran, add_list)
        self._tlaunch(tran)
        self._after_tlaunch(add_list)
        self.next.tran_last_enable_time = self.next.last_enable_times[tran]
        return self.next

    def _tlaunch(self, tran):
        places = self.p_list[tran]
        next_delay_t = self.curr.curr_delay_t.copy()
        delay_time = next_delay_t[tran]
        self.time += delay_time
        self.next.prefix += self.time
        for tran_next in range(len(self.a_matrix[0])):
            if tran_next == tran:
                if self.next.last_enable_times[tran] == -1:
                    self.next.last_enable_times[tran] = self.curr.prefix + self.time - delay_time
                continue
            if self.t_launch_is_enable[tran_next]:
                if not self._check_is_enable(tran_next):
                    self.t_launch_is_enable[tran_next] = False
                    self.curr.last_enable_times[tran_next] = -1
                else:
                    if self.next.last_enable_times[tran_next] == -1:
                        self.next.last_enable_times[tran_next] = self.next.prefix - self.before_tlaunch_timer[tran_next]
            else:
                self.next.last_enable_times[tran_next] = -1
            if self.t_launch_is_enable[tran_next]:
                next_delay_t[tran_next] -= self.before_tlaunch_timer[tran_next]
                if next_delay_t[tran_next] < 0:
                    next_delay_t[tran_next] = 0
        for place in places:
            for tran_next in self.t_list[place]:
                next_delay_t[tran_next] = self.min_delay_t[tran_next]
        self.next.curr_delay_t = next_delay_t
        self._set_next(self.next)
        self.next.curr_delay_t[tran] = self.min_delay_t[tran]

    def _before_tlaunch(self, tran, add_list):
        self.next = self.curr.clone()
        next_p_info = self.curr.nexts.get(tran)
        if next_p_info is None:
            return
        self.next.p_info = next_p_info
        self.t_launch_is_enable = self.curr.is_enable.copy()
        self.need_remove = set()
        curr_p_info = self.curr.p_info
        max_time = 0
        for i in range(len(curr_p_info)):
            get_val = self.pre[i][tran]
            add = next_p_info[i] - curr_p_info[i] + get_val
            s_t_inf = self.next.t_info[i]
            s_residence_info = self.next.residence_time_info[i]
            if get_val > 0:
                self.need_remove.add(i)
                min_val = self.next.remove_tokens(s_t_inf, get_val)
                self.next.remove_tokens(s_residence_info, get_val)
                max_time = max(max_time, min_val)
            if add > 0:
                for _ in range(add):
                    add_list.append(i)
        self.time = max_time

    def _after_tlaunch(self, add_list):
        for i in range(len(self.next.t_info)):
            if self.min_delay_p[i] == 0 and self.max_residence_time[i] == 2 ** 31 - 1:
                continue
            lst = self.next.t_info[i]
            residence_list = self.next.residence_time_info[i]
            size = len(lst)
            for _ in range(size):
                val = lst.popleft() - self.time
                val_temp = max(0, val)
                lst.append(val_temp)
                val_r = residence_list.popleft()
                if val < 0:
                    val_r -= val
                residence_list.append(val_r)
                if val_r > self.max_residence_time[i]:
                    self.next.over_max_residence_time = True
                    self.next.over_residence_time_place = i
        for i in add_list:
            self.next.t_info[i].append(self.min_delay_p[i])
            self.next.residence_time_info[i].append(0)

    def _set_next(self, next_marking):
        next_marking.is_enable = [False] * len(self.a_matrix[0])
        next_marking.nexts = {}
        for tran in range(len(self.a_matrix[0])):
            next_p_info = next_marking.p_info.copy()
            is_enable = True
            for place in range(len(self.a_matrix)):
                value = next_p_info[place] - self.pre[place][tran]
                next_p_info[place] += self.a_matrix[place][tran]
                if value < 0:
                    is_enable = False
                    break
                if self.has_capacity and self.capacity[place] < next_p_info[place]:
                    is_enable = False
                    break
            if is_enable:
                if not self.transition_flow_allowed[tran]:
                    is_enable = False
            if is_enable:
                next_marking.is_enable[tran] = True
                next_marking.nexts[tran] = next_p_info

    def _check_is_enable(self, tran):
        min_val = 2 ** 31 - 1
        places = self.p_list[tran]
        t_info = self.curr.t_info
        for place in places:
            if place in self.need_remove:
                return False
            s_t_inf = t_info[place]
            get_val = self.pre[place][tran]
            max_val = self.next.get_the_kth_small(s_t_inf, get_val)
            if max_val - self.time > 0:
                return False
            min_val = min(min_val, self.time - max_val)
        self.before_tlaunch_timer[tran] = min_val
        return True

    def enable(self, tran):
        if self.curr.over_max_residence_time:
            return False
        return self.curr.is_enable[tran]

    def get_marking(self):
        return self.curr

    def set_marking(self, marking):
        self.curr = marking

    def get_trans_count(self):
        return len(self.a_matrix[0])

    def clone(self):
        if self.has_capacity:
            return TTPPNHasResidenceTime(
                self.curr.p_info.copy(),
                self.a_matrix,
                self.pre,
                self.min_delay_p,
                self.min_delay_t,
                self.max_residence_time,
                self.capacity,
                self.place_from_places,
                self.is_resource,
            )
        return TTPPNHasResidenceTime(
            self.curr.p_info.copy(),
            self.a_matrix,
            self.pre,
            self.min_delay_p,
            self.min_delay_t,
            self.max_residence_time,
            None,
            self.place_from_places,
            self.is_resource,
        )


class TTPPNByTokenWithResTime(PetriNet):
    def __init__(self, p_info, pre, post, min_delay_p, min_delay_t, capacity, residence_time, is_resource, place_from_places, qtime_places, qtime):
        self.pre = pre
        self.post = post
        self.min_delay_p = min_delay_p
        self.min_delay_t = min_delay_t
        self.capacity = capacity
        self.residence_time = residence_time
        self.t_list = [[] for _ in range(len(pre))]
        self.p_list = [[] for _ in range(len(pre[0]))]
        self.pa_list = [[] for _ in range(len(pre[0]))]
        self.is_resource = is_resource
        self.place_from_places = place_from_places
        self.qtime_places = qtime_places
        self.qtime = qtime
        self.is_specific_get = [False] * len(pre)
        self.specific_get = [None] * len(pre)
        for place in range(len(pre)):
            for tran in range(len(pre[0])):
                if pre[place][tran] > 0:
                    self.t_list[place].append(tran)
        for tran in range(len(pre[0])):
            for place in range(len(pre)):
                if pre[place][tran] > 0:
                    self.p_list[tran].append(place)
                if post[place][tran] > 0:
                    self.pa_list[tran].append(place)
        for from_places in place_from_places:
            for place in from_places:
                self.is_specific_get[place] = True
        self.curr = TTPPNMarkingByTokenWithResTime(p_info, min_delay_t, min_delay_p)
        self._certify_enable(self.curr)

    def launch(self, tran):
        self._init()
        self._before_tlaunch(tran)
        self._tlaunch(tran)
        self._after_tlaunch(tran)
        return self.next

    def _init(self):
        self.next = self.curr.clone()
        self.before_tlaunch_timer = [0] * self.get_trans_count()
        self.get_logic_info = []
        self.get_resource_info = []
        self.need_remove = set()
        self.time = 0

    def _before_tlaunch(self, tran):
        s_p_list = self.p_list[tran]
        for place in s_p_list:
            s_t_info = self.next.t_info[place]
            self.need_remove.add(place)
            get_val = self.pre[place][tran]
            tokens = self.next.remove_tokens(s_t_info, get_val)
            if tokens:
                self.time = max(tokens[-1].timer, self.time)
            for token in tokens:
                if self.is_resource[place]:
                    if not self.is_specific_get[place]:
                        self.get_resource_info.append(token)
                    else:
                        if self.specific_get[place] is None:
                            self.specific_get[place] = []
                        self.specific_get[place].append(token)
                else:
                    self.get_logic_info.append(token)

    def _tlaunch(self, tran):
        self.time += self.next.curr_delay_t[tran]
        for t in range(self.get_trans_count()):
            if self.curr.is_enable[t] and self._check_is_enable(t):
                self.next.curr_delay_t[t] -= self.before_tlaunch_timer[t]
                self.next.curr_delay_t[t] = max(0, self.next.curr_delay_t[t])

    def _after_tlaunch(self, tran):
        for i in range(len(self.next.t_info)):
            if self.min_delay_p[i] == 0 and self.residence_time[i] == 2 ** 31 - 1:
                continue
            for token in self.next.t_info[i]:
                token.timer -= self.time
                if token.timer < 0:
                    token.timer = -token.timer
                    token.residence_time += token.timer
                    if token.residence_time > self.residence_time[i]:
                        self.next.over_max_residence_time = True
                    token.timer = 0
        self.next.prefix += self.time
        for place in range(len(self.post)):
            put = self.post[place][tran]
            for from_place in self.place_from_places[place]:
                if put == 0:
                    break
                if self.specific_get[from_place] is None:
                    continue
                if len(self.specific_get[from_place]) >= put:
                    token = None
                    for _ in range(put):
                        last = self.specific_get[from_place].pop()
                        token = Token(last.get_id(), self.min_delay_p[place], 0)
                    if token is not None:
                        self.next.t_info[place].append(token)
                    put = 0
                else:
                    token = None
                    size = len(self.specific_get[from_place])
                    for _ in range(size):
                        last = self.specific_get[from_place].pop()
                        token = Token(last.get_id(), self.min_delay_p[place], 0)
                    if token is not None:
                        self.next.t_info[place].append(token)
                    put -= size
            for _ in range(put):
                token = None
                if self.is_resource[place]:
                    if self.get_resource_info:
                        last = self.get_resource_info.pop()
                        token = Token(last.get_id(), self.min_delay_p[place], 0)
                        self.next.over_max_residence_time = self.qtime_places[place] and self._is_over_qtime(token.get_id())
                    else:
                        token = Token(self.next.max_id, self.min_delay_p[place], 0)
                        self.next.max_id += 1
                        self.next.over_max_residence_time = self.qtime_places[place] and self._is_over_qtime(token.get_id())
                else:
                    if self.get_logic_info:
                        last = self.get_logic_info.pop()
                        token = Token(last.get_id(), self.min_delay_p[place], 0)
                    else:
                        token = Token(self.next.max_id, self.min_delay_p[place], 0)
                        self.next.max_id += 1
                self.next.t_info[place].append(token)
        for place in self.p_list[tran]:
            for t in self.t_list[place]:
                self.next.curr_delay_t[t] = self.min_delay_t[t]
        self.next.curr_delay_t[tran] = self.min_delay_t[tran]
        self._certify_enable(self.next)

    def _is_over_qtime(self, token_id):
        if token_id not in self.next.qtime_map:
            self.next.qtime_map[token_id] = self.next.prefix
            return False
        time = self.next.prefix - self.next.qtime_map[token_id]
        self.next.qtime_map[token_id] = self.next.get_prefix()
        return time > self.qtime

    def _check_is_enable(self, tran):
        s_p_list = self.p_list[tran]
        time = 0
        for place in s_p_list:
            s_t_info = self.next.t_info[place]
            if place in self.need_remove:
                return False
            get_val = self.pre[place][tran]
            time = max(self.next.get_the_kth_small(s_t_info, get_val), time)
        if time < self.time:
            self.before_tlaunch_timer[tran] = self.time - time
            return True
        return False

    def _certify_enable(self, marking):
        marking.is_enable = [False] * len(self.pre[0])
        p_info = marking.get_p_info()
        for tran in range(len(self.pre[0])):
            enable = True
            for place in range(len(self.pre)):
                s_info = marking.t_info[place]
                value = p_info[place] - self.pre[place][tran]
                if value < 0:
                    enable = False
                    break
                if p_info[place] > self.capacity[place]:
                    enable = False
                    break
                for token in s_info:
                    if token.residence_time > self.residence_time[place]:
                        enable = False
                        break
                if not enable:
                    break
            marking.is_enable[tran] = enable

    def enable(self, tran):
        return (not self.curr.over_max_residence_time) and self.curr.is_enable[tran]

    def get_marking(self):
        return self.curr

    def set_marking(self, marking):
        self.curr = marking

    def get_trans_count(self):
        return len(self.pre[0])

    def clone(self):
        return TTPPNByTokenWithResTime(self.curr.get_p_info(), self.pre, self.post, self.min_delay_p, self.min_delay_t, self.capacity, self.residence_time, self.is_resource, self.place_from_places, self.qtime_places, self.qtime)


class TTimePetriNet(PetriNet):
    def __init__(self, p_info, a_matrix, min_delay_t):
        self.curr = TTimeMarking(p_info, 0, min_delay_t.copy())
        self.a_matrix = a_matrix
        self.min_delay_t = min_delay_t
        self.t_list = [[] for _ in range(len(a_matrix))]
        self.p_list = [[] for _ in range(len(a_matrix[0]))]
        for place in range(len(a_matrix)):
            for tran in range(len(a_matrix[0])):
                if a_matrix[place][tran] < 0:
                    self.t_list[place].append(tran)
        for tran in range(len(a_matrix[0])):
            for place in range(len(a_matrix)):
                if a_matrix[place][tran] < 0:
                    self.p_list[tran].append(place)
        self._set_next(self.curr)

    def launch(self, tran):
        places = self.p_list[tran]
        next_p_info = self.curr.nexts.get(tran)
        next_delay_t = self.curr.curr_delay_t.copy()
        delay_time = next_delay_t[tran]
        for tran_next in range(len(self.a_matrix[0])):
            if self.curr.is_enable[tran_next]:
                next_delay_t[tran_next] -= delay_time
                if next_delay_t[tran_next] < 0:
                    next_delay_t[tran_next] = 0
        for place in places:
            for tran_next in self.t_list[place]:
                next_delay_t[tran_next] = self.min_delay_t[tran_next]
        next_delay_t[tran] = self.min_delay_t[tran]
        next_marking = TTimeMarking(next_p_info, self.curr.prefix + delay_time, next_delay_t)
        self._set_next(next_marking)
        return next_marking

    def _set_next(self, next_marking):
        next_marking.nexts = {}
        for tran in range(len(self.a_matrix[0])):
            next_p_info = next_marking.p_info.copy()
            is_enable = True
            for place in range(len(self.a_matrix)):
                next_p_info[place] += self.a_matrix[place][tran]
                if next_p_info[place] < 0:
                    is_enable = False
                    break
            if is_enable:
                next_marking.is_enable[tran] = True
                next_marking.nexts[tran] = next_p_info

    def enable(self, tran):
        return self.curr.is_enable[tran]

    def get_marking(self):
        return self.curr

    def set_marking(self, marking):
        self.curr = marking

    def get_trans_count(self):
        return len(self.a_matrix[0])

    def clone(self):
        return TTimePetriNet(self.curr.p_info.copy(), self.a_matrix, self.min_delay_t)

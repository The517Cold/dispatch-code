from collections import deque
from .abstract_search import AbstractSearch
from ..utils.result import Result
from ..utils.tabu_condition import TabuCondition
from ..utils.tran_order_entity import TranOrderEntity
from ..architecture import HasResideceTime


class AbstractGreedy(AbstractSearch):
    def __init__(self, petri_net, end):
        super().__init__()
        self.trans = []
        self.markings = []
        self.petri_net = petri_net
        self.end = end
        self.is_over_residence_time = False
        self.is_cut_point = False
        self.seen = {}
        self.branch_point = {}
        self.branch_point_info = None
        self.extra_info = {}
        self.extend_marking_count = 0
        self.place = 0
        self.token = 0
        self.tran_marking = {}
        self.comparator = None

    def f(self, tran):
        raise NotImplementedError()

    def search(self):
        self.markings.append(self.petri_net.get_marking())
        self._find()
        self.extra_info["extendMarkingCount"] = self.extend_marking_count
        return Result(self.trans, self.markings)

    def get_extra_info(self):
        return self.extra_info

    def _find(self):
        self.extend_marking_count += 1
        curr = self.markings[-1]
        self._save_new_marking(curr)
        if self.same(curr):
            return True
        tran_marking_temp = self.tran_marking
        queue = self._sort_trans()
        for tran in queue:
            next_marking = self.tran_marking[tran]
            if not self._is_need_extend(next_marking):
                continue
            self._extend(next_marking, tran)
            if self._find():
                return True
            self._back(curr)
        self.tran_marking = tran_marking_temp
        return False

    def _sort_trans(self):
        self.tran_marking = {}
        queue = []
        for tran in range(self.petri_net.get_trans_count()):
            if self.petri_net.enable(tran):
                self.tran_marking[tran] = self.petri_net.launch(tran)
                queue.append(tran)
        if self.comparator is not None:
            queue.sort(key=self.comparator)
        else:
            queue.sort(key=lambda x: self.f(x))
        self.sort_again(queue)
        return queue

    def sort_again(self, queue):
        return

    def _save_new_marking(self, curr):
        if curr not in self.seen:
            self.seen[curr] = TabuCondition()

    def _is_need_extend(self, next_marking):
        if next_marking in self.seen:
            tabu_condition = self.seen[next_marking]
            return tabu_condition.judge(next_marking)
        return True

    def _extend(self, next_marking, tran):
        curr = self.petri_net.get_marking()
        if self.is_over_residence_time:
            val = self.branch_point.get(curr, {})
            val[self.place] = self.token
            self.branch_point[curr] = val
        self.petri_net.set_marking(next_marking)
        self.trans.append(tran)
        self.markings.append(next_marking)
        self.is_over_residence_time = False
        self.is_cut_point = False
        self.branch_point_info = None
        if isinstance(next_marking, HasResideceTime) and next_marking.is_over_residece_time():
            self.is_over_residence_time = True

    def _back(self, curr):
        next_marking = self.petri_net.get_marking()
        self.petri_net.set_marking(curr)
        self.trans.pop()
        self.markings.pop()
        tabu_condition = self.seen[next_marking]
        tabu_condition.set_over_residece_time(self.is_over_residence_time)
        if self.is_over_residence_time:
            has_residece_time = next_marking
            if self.branch_point_info is None:
                if has_residece_time.is_over_residece_time():
                    self.place = has_residece_time.get_over_residence_time_place()
                    self.token = 0
                self.token += max(curr.get_p_info()[self.place] - next_marking.get_p_info()[self.place], 0)
                time_val = has_residece_time.get_residence_time(self.place, self.token)
                tabu_condition.renew(self.place, self.token, time_val)
            else:
                for place in list(self.branch_point_info.keys()):
                    token = self.branch_point_info[place]
                    token += max(curr.get_p_info()[place] - next_marking.get_p_info()[place], 0)
                    self.branch_point_info[place] = token
                    time_val = has_residece_time.get_residence_time(place, token)
                    tabu_condition.renew(place, token, time_val)
        if not self.is_over_residence_time and curr in self.branch_point:
            self.is_over_residence_time = True
            self.is_cut_point = True
            self.branch_point_info = self.branch_point[curr]


class Greedy(AbstractGreedy):
    def __init__(self, petri_net, end):
        super().__init__(petri_net, end)

    def f(self, tran):
        return self.tran_marking[tran].get_prefix()


class GreedySRPT(AbstractGreedy):
    def __init__(self, petri_net, end):
        super().__init__(petri_net, end)

    def f(self, tran):
        has_residece_time = self.tran_marking[tran]
        waste_time = 0
        last_time = 0
        for i in range(1, len(has_residece_time.get_p_info())):
            waste_time += has_residece_time.get_residence_time(i)
        for i in range(1, len(has_residece_time.get_p_info())):
            last_time += has_residece_time.get_time(i)
        rate = 0.5
        return int(rate * last_time - (1 - rate) * waste_time)


class GreedyWithDepth(Greedy):
    def __init__(self, petri_net, end, launch_immediate, n):
        super().__init__(petri_net, end)
        self.priors = []
        self.find = set()
        self.times = [0] * petri_net.get_trans_count()
        for tran_order_entity in launch_immediate:
            if tran_order_entity not in self.find:
                lst = [tran_order_entity]
                self._find_prior_path(lst, tran_order_entity, launch_immediate)
        self.n = n
        self.comparator = self._comparator()
        self.pre_len = 0
        self.back_tran = 0

    def _find_prior_path(self, trans, tran_order_entity, launch_immediate):
        self.find.add(tran_order_entity)
        if tran_order_entity not in launch_immediate:
            mp = {}
            value = 0
            for i in trans:
                mp[i] = value
                value += 1
            self.priors.append(mp)
            return
        for nxt in launch_immediate[tran_order_entity]:
            trans.append(nxt)
            self._find_prior_path(trans, nxt, launch_immediate)
            trans.pop()

    def _comparator(self):
        def compare(tran):
            return self._find_min_tran(tran)
        return compare

    def _find_min_tran(self, tran):
        end_marking_time = 2 ** 31 - 1
        curr = self.petri_net.get_marking()
        min_val = 2 ** 31 - 1
        count = 1
        queue = deque()
        if self.same(self.tran_marking[tran]):
            return self.tran_marking[tran].get_prefix()
        queue.append(self.tran_marking[tran])
        while count < self.n and queue:
            length = len(queue)
            for _ in range(length):
                self.petri_net.set_marking(queue.popleft())
                for tran_next in range(self.petri_net.get_trans_count()):
                    if self.petri_net.enable(tran_next):
                        next_marking = self.petri_net.launch(tran_next)
                        if self.same(next_marking):
                            end_marking_time = min(end_marking_time, next_marking.get_prefix())
                        else:
                            queue.append(next_marking)
            count += 1
        while queue:
            min_val = min(min_val, queue.popleft().get_prefix())
        self.petri_net.set_marking(curr)
        if end_marking_time != 2 ** 31 - 1:
            return end_marking_time // 10 if end_marking_time < min_val else min_val
        return min_val

    def sort_again(self, queue):
        if self.pre_len < len(self.trans):
            self.times[self.trans[-1]] += 1
        elif self.pre_len < len(self.trans):
            self.times[self.back_tran] -= 1
        if self.trans:
            self.back_tran = self.trans[-1]
        for mp in self.priors:
            lst = []
            for tran in queue:
                tran_order_entity = TranOrderEntity(tran, self.times[tran] + 1)
                if tran_order_entity in mp:
                    lst.append(tran_order_entity)
            lst.sort(key=lambda x: mp[x], reverse=True)
            for i in range(len(queue)):
                tran_order_entity = TranOrderEntity(queue[i], self.times[queue[i]] + 1)
                if tran_order_entity in mp:
                    queue[i] = lst[-1].get_tran()
                    lst.pop()


class GreedyWithGA(Greedy):
    def __init__(self, petri_net, end, priorities):
        super().__init__(petri_net, end)
        self.priorities = priorities
        self.comparator = self._comparator()

    def set_priorities(self, priorities):
        self.priorities = priorities

    def get_priorities(self):
        return self.priorities

    def _comparator(self):
        def compare(tran):
            return (-self.priorities[tran], self.f(tran))
        return compare

    def f(self, tran):
        return self.tran_marking[tran].get_prefix()

import random
from .abstract_search import AbstractSearch
from ..utils.result import Result


class LowSpaceDfs(AbstractSearch):
    def __init__(self, petri_net, end):
        super().__init__()
        self.petri_net = petri_net
        self.end = end
        self.min_length = 2 ** 31 - 1
        self.trans = []
        self.markings = []
        self.seen = set()

    def search(self):
        curr = self.petri_net.get_marking()
        self._dfs_to_find_min_length(0)
        self.petri_net.set_marking(curr)
        if self._dfs_to_find_best_path(0):
            return Result(self.trans, self.markings)
        return None

    def get_extra_info(self):
        return None

    def _dfs_to_find_min_length(self, length):
        curr = self.petri_net.get_marking()
        if self.same(curr):
            self.min_length = min(self.min_length, length)
        for tran in range(self.petri_net.get_trans_count()):
            if self.petri_net.enable(tran):
                next_marking = self.petri_net.launch(tran)
                if next_marking in self.seen:
                    continue
                self.seen.add(next_marking)
                self.petri_net.set_marking(next_marking)
                self._dfs_to_find_min_length(next_marking.get_prefix())
                self.seen.remove(next_marking)
                self.petri_net.set_marking(curr)

    def _dfs_to_find_best_path(self, length):
        curr = self.petri_net.get_marking()
        if self.same(curr) and length == self.min_length:
            return True
        if length >= self.min_length:
            return False
        for tran in range(self.petri_net.get_trans_count()):
            if self.petri_net.enable(tran):
                next_marking = self.petri_net.launch(tran)
                if next_marking in self.seen:
                    continue
                self.trans.append(tran)
                self.markings.append(next_marking)
                self.seen.add(next_marking)
                self.petri_net.set_marking(next_marking)
                if self._dfs_to_find_best_path(next_marking.get_prefix()):
                    return True
                self.trans.pop()
                self.markings.pop()
                self.seen.remove(next_marking)
                self.petri_net.set_marking(curr)
        return False


class RandomDfs(AbstractSearch):
    def __init__(self, petri_net, end):
        super().__init__()
        self.petri_net = petri_net
        self.end = end
        self.seen = set()
        self.trans = []
        self.markings = []
        self.extra_info = {}
        self.extend_marking_count = 0
        curr = petri_net.get_marking()
        self.markings.append(curr)

    def search(self):
        self._find()
        return Result(self.trans, self.markings)

    def get_extra_info(self):
        self.extra_info["extendMarkingCount"] = self.extend_marking_count
        return self.extra_info

    def _find(self):
        curr = self.petri_net.get_marking()
        while not self.same(curr):
            tran = self._random_chose()
            while tran == -1:
                if not self.trans:
                    return
                self.markings.pop()
                self.trans.pop()
                curr = self.markings[-1]
                self.petri_net.set_marking(curr)
                tran = self._random_chose()
            next_marking = self.petri_net.launch(tran)
            self.extend_marking_count += 1
            self.petri_net.set_marking(next_marking)
            self.seen.add(next_marking)
            self.markings.append(next_marking)
            self.trans.append(tran)
            curr = next_marking

    def _random_chose(self):
        enable_trans = []
        for tran in range(self.petri_net.get_trans_count()):
            if self.petri_net.enable(tran):
                next_marking = self.petri_net.launch(tran)
                if next_marking not in self.seen:
                    enable_trans.append(tran)
        if not enable_trans:
            return -1
        idx = random.randint(0, len(enable_trans) - 1)
        return enable_trans[idx]

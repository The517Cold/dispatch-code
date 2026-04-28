from collections import deque


class ReachabilityGraphBulider:
    def __init__(self, petri_net):
        self.petri_net = petri_net
        self.reachability_graph = {petri_net.get_marking(): []}

    def make_reachability_graph(self):
        queue = deque()
        queue.append(self.petri_net.get_marking())
        while queue:
            curr = queue.popleft()
            trans = self.reachability_graph.get(curr, [])
            self.petri_net.set_marking(curr)
            for tran in range(self.petri_net.get_trans_count()):
                if self.petri_net.enable(tran):
                    trans.append(tran)
                    next_marking = self.petri_net.launch(tran)
                    if next_marking in self.reachability_graph:
                        continue
                    self.reachability_graph[next_marking] = []
                    queue.append(next_marking)
        return self.reachability_graph

    def is_reach(self, end):
        if len(self.reachability_graph) <= 1:
            return self._has_not_reachability_graph(end)
        return self._has_reachability_graph(end)

    def _has_reachability_graph(self, end):
        for marking in self.reachability_graph:
            if self._is_same(end, marking):
                return True
        return False

    def _has_not_reachability_graph(self, end):
        queue = deque()
        queue.append(self.petri_net.get_marking())
        while queue:
            curr = queue.popleft()
            if self._is_same(end, curr):
                return True
            trans = self.reachability_graph.get(curr, [])
            self.petri_net.set_marking(curr)
            for tran in range(self.petri_net.get_trans_count()):
                if self.petri_net.enable(tran):
                    trans.append(tran)
                    next_marking = self.petri_net.launch(tran)
                    if next_marking in self.reachability_graph:
                        continue
                    self.reachability_graph[next_marking] = []
                    queue.append(next_marking)
        return False

    def _is_same(self, end, curr):
        p_info = curr.get_p_info()
        for i in range(len(p_info)):
            if end[i] == -1:
                continue
            if p_info[i] != end[i]:
                return False
        return True

    def get_reachable_markings_count(self):
        if not self.reachability_graph:
            self.make_reachability_graph()
        return len(self.reachability_graph)

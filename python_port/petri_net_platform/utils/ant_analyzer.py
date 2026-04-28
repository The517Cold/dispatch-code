from .ant_analyze_result import AntAnalyzeResult
from ..search.ant import AntClonyOptimization


class AntAnalyzer:
    def __init__(self, search, petri_net):
        if isinstance(search, AntClonyOptimization):
            self.pheromone_map = search.get_pheromone_map()
            self.petri_net = petri_net.clone()
        else:
            raise Exception("不是蚁群算法,无法分析")

    def analyze(self, trans):
        pheromones = []
        curr = self.petri_net.get_marking()
        pheromone = self.pheromone_map.get(curr)
        pheromones.append(pheromone)
        for i in range(len(trans) - 1):
            tran = trans[i]
            if not self.petri_net.enable(tran):
                return None
            curr = self.petri_net.launch(tran)
            if curr not in self.pheromone_map:
                return None
            pheromones.append(self.pheromone_map.get(curr))
            self.petri_net.set_marking(curr)
        return AntAnalyzeResult(pheromones, trans)

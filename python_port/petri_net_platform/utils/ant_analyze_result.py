class AntAnalyzeResult:
    def __init__(self, pheromones, trans):
        self.pheromones = pheromones
        self.trans = trans

    def __str__(self):
        lines = []
        for i in range(len(self.pheromones)):
            pheromone = self.pheromones[i]
            tran = self.trans[i]
            lines.append("第" + str(i) + "个标识,下一个发射的变迁号为: " + str(tran) + "\n" + "使能变迁信息素信息:")
            nexts = pheromone.get_next_ps()
            next_trans_list = sorted(nexts.keys())
            for next_tran in next_trans_list:
                lines.append("第" + str(next_tran) + "号变迁的信息素为: " + str(nexts[next_tran]))
        return "\n".join(lines)

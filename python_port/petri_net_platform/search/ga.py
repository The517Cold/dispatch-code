import random
import time
from .greedy import GreedyWithGA
from ..utils.result import Result
from ..utils.low_space_link import LowSpaceLink, TranLink
from ..utils.result import Result
from ..architecture import Search
from .ant import AntClonyOptimization


class Gene:
    def __init__(self):
        self.prefix = 0

    def get_prefix(self):
        return self.prefix

    def set_prefix(self, prefix):
        self.prefix = prefix


class GeneWithPriorities(Gene):
    def __init__(self, priorities):
        super().__init__()
        self.priorities = priorities
        self.prefix = (2 ** 31 - 1) // 2

    def get_priorities(self):
        return self.priorities

    def set_priorities(self, priorities):
        self.priorities = priorities


class GeneWithTrans(Gene):
    def __init__(self):
        super().__init__()
        self.trans = []
        self.markings = []

    def get_trans(self):
        return self.trans

    def set_trans(self, trans):
        self.trans = trans

    def get_markings(self):
        return self.markings

    def set_markings(self, markings):
        self.markings = markings


class MiddleFinder:
    @staticmethod
    def find_middle(genes):
        genes.sort(key=lambda g: g.get_prefix())
        return genes[len(genes) >> 1].get_prefix()


class GA(Search):
    def __init__(self, petri_net, end, n, init_info, genes_count):
        self.petri_net = petri_net
        self.end = end
        self.n = n
        self.genes_count = genes_count
        self.var_counts = 0
        self.new_genes = []
        self.genes = []
        self.result = None
        self.extra_info = {}
        self.on_iteration = None
        for _ in range(genes_count):
            self.genes.append(self.init_gene(init_info))
        for gene in self.genes:
            self.use_gene(gene)

    def search(self):
        for idx in range(self.n):
            start = time.perf_counter()
            self.eliminate()
            self.var_counts = 300
            for _ in range(self.var_counts):
                self.variate(self.choose_gene())
            self.cross()
            makespan = min((gene.get_prefix() for gene in self.genes), default=2 ** 31 - 1)
            elapsed = time.perf_counter() - start
            if self.on_iteration is not None:
                self.on_iteration(idx + 1, self.n, elapsed, makespan)
        self.looking_the_best_gene()
        return self.result

    def choose_gene(self):
        idx = random.randint(0, len(self.genes) - 1)
        return self.genes[idx]

    def eliminate(self):
        temp = []
        mid_value = MiddleFinder.find_middle(self.genes)
        for gene in self.genes:
            if gene.get_prefix() <= mid_value:
                temp.append(gene)
        self.genes = temp

    def cross(self):
        self.new_genes = []
        for _ in range(len(self.genes), self.genes_count):
            idx1 = random.randint(0, len(self.genes) - 1)
            idx2 = random.randint(0, len(self.genes) - 1)
            father = self.genes[idx1]
            mother = self.genes[idx2]
            child = self.make_new_gene(father, mother)
            self.new_genes.append(child)
        self.calculate()

    def calculate(self):
        for gene in self.new_genes:
            self.use_gene(gene)
        self.genes.extend(self.new_genes)

    def looking_the_best_gene(self):
        best_gene = None
        best_value = 2 ** 31 - 1
        for gene in self.genes:
            value = gene.get_prefix()
            if best_value > value:
                best_value = value
                best_gene = gene
        self.result = self.make_result(best_gene)

    def make_new_gene(self, father, mother):
        raise NotImplementedError()

    def use_gene(self, gene):
        raise NotImplementedError()

    def variate(self, gene):
        raise NotImplementedError()

    def init_gene(self, init_info):
        raise NotImplementedError()

    def make_result(self, best_gene):
        raise NotImplementedError()

    def get_extra_info(self):
        return self.extra_info


class GAWithPriorities(GA):
    def __init__(self, petri_net, end, n, proportion, genes_count):
        super().__init__(petri_net, end, n, proportion, genes_count)

    def use_gene(self, gene):
        gene_with_priorities = gene
        search = GreedyWithGA(self.petri_net.clone(), self.end, gene_with_priorities.get_priorities())
        result = search.search()
        size = len(result.get_markings())
        gene.set_prefix(result.get_markings()[size - 1].get_prefix())

    def variate(self, gene):
        return

    def make_new_gene(self, father, mother):
        priorities1 = father.get_priorities()
        priorities2 = mother.get_priorities()
        priorities = [0.0] * self.petri_net.get_trans_count()
        cutpoint = random.randint(0, len(priorities) - 1)
        for j in range(0, cutpoint):
            priorities[j] = priorities1[j]
        for j in range(cutpoint, len(priorities)):
            priorities[j] = priorities2[j]
        return GeneWithPriorities(priorities)

    def make_result(self, best_gene):
        best_priorities = best_gene.get_priorities()
        return GreedyWithGA(self.petri_net, self.end, best_priorities).search()

    def init_gene(self, init_info):
        proportion = init_info
        priorities = [0.0] * self.petri_net.get_trans_count()
        for i in range(self.petri_net.get_trans_count()):
            priorities[i] = random.random()
            if priorities[i] < proportion:
                priorities[i] = -1.0
        return GeneWithPriorities(priorities)


class GAWithTrans(GA):
    def __init__(self, petri_net, end, n, init_info, genes_count):
        self.search_engine = None
        super().__init__(petri_net, end, n, init_info, genes_count)

    def make_new_gene(self, father, mother):
        father1 = father
        mother1 = mother
        hash_map = {}
        for i in range(len(father1.get_markings())):
            marking = father1.get_markings()[i]
            hash_map[marking] = i
        mapping = {}
        for i in range(1, len(mother1.get_markings())):
            marking = mother1.get_markings()[i]
            if marking in hash_map:
                father_idx = hash_map[marking] - 1
                mother_idx = i - 1
                length = father_idx + (len(mother1.get_trans()) - mother_idx)
                trans = []
                for j in range(father_idx + 1):
                    trans.append(father1.get_trans()[j])
                for j in range(mother_idx + 1, len(mother1.get_trans())):
                    trans.append(mother1.get_trans()[j])
                gene_with_trans = GeneWithTrans()
                gene_with_trans.set_trans(trans)
                value = father1.get_markings()[father_idx].get_prefix() + mother1.get_prefix() - mother1.get_markings()[mother_idx].get_prefix()
                mapping[gene_with_trans] = value
        return self.choose(mapping)

    def variate(self, gene):
        gene_with_trans = gene
        if not gene_with_trans.get_trans():
            return
        windows_len = random.randint(0, len(gene_with_trans.get_trans()) - 1)
        for head in range(len(gene_with_trans.get_trans()) - windows_len):
            rear = head + windows_len
            if self.check(gene_with_trans, head, rear) and random.random() < 0.1:
                trans = gene_with_trans.get_trans()
                trans[head], trans[rear] = trans[rear], trans[head]
                self.use_gene(gene_with_trans)

    def check(self, gene_with_trans, head, rear):
        curr = self.petri_net.get_marking()
        start = gene_with_trans.get_markings()[head]
        self.petri_net.set_marking(start)
        if not self.petri_net.enable(gene_with_trans.get_trans()[rear]):
            self.petri_net.set_marking(curr)
            return False
        self.petri_net.set_marking(self.petri_net.launch(gene_with_trans.get_trans()[rear]))
        for i in range(head + 1, rear):
            if not self.petri_net.enable(gene_with_trans.get_trans()[i]):
                self.petri_net.set_marking(curr)
                return False
            self.petri_net.set_marking(self.petri_net.launch(gene_with_trans.get_trans()[i]))
        res = self.petri_net.enable(gene_with_trans.get_trans()[head])
        self.petri_net.set_marking(curr)
        return res

    def choose(self, mapping):
        if not mapping:
            return None
        sum_val = 0.0
        sum1 = 0.0
        for prefix in mapping.values():
            sum_val += prefix
        for prefix in mapping.values():
            sum1 += sum_val / prefix
        sum_val *= 100
        random_val = random.random() * sum1
        total = 0.0
        for gene_with_trans in mapping:
            total += sum_val / mapping[gene_with_trans]
            if total - random_val >= 1:
                return gene_with_trans
        return None

    def use_gene(self, gene):
        gene_with_trans = gene
        trans = gene_with_trans.get_trans()
        petri_net1 = self.petri_net.clone()
        markings = [petri_net1.get_marking()]
        curr = None
        for tran in trans:
            curr = petri_net1.launch(tran)
            markings.append(curr)
            petri_net1.set_marking(curr)
        if curr is not None:
            gene_with_trans.set_prefix(curr.get_prefix())
        gene_with_trans.set_markings(markings)

    def init_gene(self, init_info):
        if self.search_engine is None:
            self.search_engine = AntClonyOptimization(init_info, 10, self.petri_net.clone(), self.end)
            self.search_engine.search()
        ants = self.search_engine.get_ants()
        ant = ants.pop()
        best_ant = self.search_engine.get_best_ant()
        if not ant.is_same():
            result = self.make_result_from_ant(best_ant)
        else:
            result = self.make_result_from_ant(ant)
        gene_with_trans = GeneWithTrans()
        gene_with_trans.set_trans(result.get_trans())
        return gene_with_trans

    def make_result(self, best_gene):
        gene_with_trans = best_gene
        trans = gene_with_trans.get_trans()
        markings = gene_with_trans.get_markings()
        return Result(trans, markings)

    def make_result_from_ant(self, ant):
        if ant is None:
            return None
        trans = []
        markings = [self.petri_net.get_marking()]
        log = ant.get_log()
        for log_entity in log:
            trans.append(log_entity.get_tran())
            markings.append(log_entity.get_next())
        return Result(trans, markings)

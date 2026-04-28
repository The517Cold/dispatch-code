import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from ..function_switch import SearchFunctionSwitch
from ..variable import SearchVariable
from ..utils.result import Result


class LogEntity:
    def __init__(self, curr, next_marking, tran, pheromone):
        self.curr = curr
        self.next = next_marking
        self.tran = tran
        self.pheromone = pheromone

    def get_curr(self):
        return self.curr

    def get_next(self):
        return self.next

    def get_tran(self):
        return self.tran

    def get_pheromone(self):
        return self.pheromone


class Pheromone:
    def __init__(self, petri_net=None, next_ps=None):
        if next_ps is not None:
            self.next_ps = next_ps
            return
        self.next_ps = {}
        curr = petri_net.get_marking()
        for i in range(petri_net.get_trans_count()):
            if petri_net.enable(i):
                next_marking = petri_net.launch(i)
                if SearchFunctionSwitch.isGreedy:
                    add = self._greedy_strategy(next_marking, curr)
                else:
                    add = self._normal_strategy()
                self.next_ps[i] = add

    def get_next_ps(self):
        return self.next_ps

    def _greedy_strategy(self, next_marking, curr):
        div = next_marking.get_prefix() - curr.get_prefix()
        div = 1 if div == 0 else div
        add = (SearchVariable.initPheromone * 100) // div
        return 1 if add == 0 else add

    def _normal_strategy(self):
        return SearchVariable.initPheromone

    def has_next(self, tabu, global_tabu, petri_net):
        for tran in self.next_ps:
            if petri_net.enable(tran):
                marking = petri_net.launch(tran)
                if marking not in tabu and marking not in global_tabu:
                    return True
        return False

    def dilute(self):
        for i in list(self.next_ps.keys()):
            value = self.next_ps[i]
            value = int(value * SearchVariable.diluteRate)
            value = 1 if value == 0 else value
            self.next_ps[i] = value

    def add(self, i, add_val):
        value = self.next_ps.get(i, 0)
        value += add_val
        self.next_ps[i] = value

    def chose(self, tabu, global_tabu, petri_net):
        total = 0
        temp = {}
        for tran in self.next_ps:
            marking = petri_net.launch(tran)
            if marking in tabu or marking in global_tabu:
                continue
            total += self.next_ps[tran]
            temp[tran] = self.next_ps[tran]
        if total == 0:
            return -1
        rand = int(random.random() * total)
        total = 0
        for tran in temp:
            total += temp[tran]
            if total >= rand:
                return tran
        return -1

    def get_next_marking(self, tran, petri_net):
        return petri_net.launch(tran)

    def clone(self):
        return Pheromone(next_ps=dict(self.next_ps))

    def __eq__(self, other):
        if not isinstance(other, Pheromone):
            return False
        return self.next_ps == other.next_ps

    def __hash__(self):
        return hash(tuple(sorted(self.next_ps.items())))


class Ant:
    def __init__(self, map_data, global_tabu, petri_net, end):
        self.log = []
        self.tabu = set()
        self.global_tabu = global_tabu
        self.start = petri_net.get_marking()
        self.map = map_data
        self.pheromone = map_data.get(petri_net.get_marking())
        self.petri_net = petri_net
        self.end = end
        self.step = 0

    def get_step(self):
        return self.step

    def set_map(self, map_data):
        self.map = map_data

    def go_back_home(self):
        self.petri_net.set_marking(self.start)
        self.tabu = set()
        self.log = []
        self.step = 0

    def get_log(self):
        return self.log

    def next(self):
        curr = self.petri_net.get_marking()
        tran = self.pheromone.chose(self.tabu, self.global_tabu, self.petri_net)
        next_marking = self.petri_net.launch(tran)
        self.tabu.add(next_marking)
        self.petri_net.set_marking(next_marking)
        self.log.append(LogEntity(curr, next_marking, tran, self.pheromone))
        self.step += 1

    def travel(self):
        while not self.is_same() and not self.is_dead():
            self.next()

    def is_same(self):
        curr = self.petri_net.get_marking()
        p_info = curr.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            if p_info[i] != self.end[i]:
                return False
        return True

    def is_dead(self):
        curr = self.petri_net.get_marking()
        self.pheromone = self.map.get(curr, Pheromone(self.petri_net))
        if not self.pheromone.has_next(self.tabu, self.global_tabu, self.petri_net):
            return True
        return False

    def get_tabu(self):
        return self.tabu

    def clone(self):
        log_temp = []
        for log_entity in self.log:
            log_temp.append(LogEntity(log_entity.get_curr(), log_entity.get_next(), log_entity.get_tran(), self.pheromone))
        ant = Ant.__new__(Ant)
        ant.log = log_temp
        return ant


class AntWithMemory(Ant):
    def __init__(self, map_data, global_tabu, petri_net, end):
        super().__init__(map_data, global_tabu, petri_net, end)
        self.bad_markings = []
        self.branch_point = set()
        self.life = 0
        self.is_over_residence_time = False

    def get_bad_markings(self):
        return self.bad_markings

    def travel(self):
        self.clean()
        self.trace_back_strategy()

    def clean(self):
        self.bad_markings = []
        self.branch_point = set()
        self.life = 0

    def trace_back_strategy(self):
        if SearchFunctionSwitch.isMinStepStrategy:
            self.min_step_strategy()
        elif SearchFunctionSwitch.isMinFallStrategy:
            self.min_fall_strategy()
        SearchVariable.life = SearchVariable.life * 0.5707 + self.life * (1 - 0.5707)
        SearchVariable.lifeVariance = SearchVariable.lifeVariance * 0.5707 + (SearchVariable.life - self.life) * (SearchVariable.life - self.life) * (1 - 0.5707)
        SearchVariable.lifeUpperLimit = max(100, SearchVariable.life + 3 * (SearchVariable.lifeVariance ** 0.5))

    def min_step_strategy(self):
        while not self.is_same() and self.life < SearchVariable.lifeUpperLimit:
            self.life += 1
            while self.is_dead():
                if not self.back_one_step():
                    return
            self.renew_over_residence_time_flag()
            self.next()

    def min_fall_strategy(self):
        while not self.is_same() and self.life < SearchVariable.lifeUpperLimit:
            while self.is_dead():
                self.check_is_over_residence_time()
                if not self.back_one_step():
                    return
                self.life += 1
            self.renew_over_residence_time_flag()
            self.next()

    def check_is_over_residence_time(self):
        marking = self.petri_net.get_marking()
        if isinstance(marking, object) and hasattr(marking, "is_over_residece_time"):
            if marking.is_over_residece_time():
                self.is_over_residence_time = True

    def back_one_step(self):
        if not self.log:
            return False
        if not self.is_over_residence_time and self.petri_net.get_marking() in self.branch_point:
            self.is_over_residence_time = True
        if not self.is_over_residence_time:
            self.bad_markings.append(self.petri_net.get_marking())
        curr = self.log[-1].get_curr()
        self.log.pop()
        self.petri_net.set_marking(curr)
        self.pheromone = self.map.get(curr, Pheromone(self.petri_net))
        return True

    def renew_over_residence_time_flag(self):
        if self.is_over_residence_time:
            self.branch_point.add(self.petri_net.get_marking())
        self.is_over_residence_time = False


class AntClonyOptimization:
    default_on_round = None
    def __init__(self, ant_count, round_count, petri_net, end):
        self.init(ant_count, round_count, petri_net, end)

    def init(self, ant_count, round_count, petri_net, end):
        self.ant_count = ant_count
        self.round = round_count
        self.petri_net = petri_net
        self.global_tabu = set()
        self.end = end
        self.i = 0
        self.round_rate = [idx * 1.0 / round_count for idx in range(round_count)]
        self.extra_info = {}
        self.ans_is_change = False
        self.unchange_count = 0
        pheromone = Pheromone(petri_net)
        marking = petri_net.get_marking()
        self.map = {marking: pheromone}
        self.map_temp = {}
        self.ants = []
        self.on_round = AntClonyOptimization.default_on_round
        if SearchFunctionSwitch.isAntHasMemory:
            for _ in range(ant_count):
                self.ants.append(AntWithMemory(self.map, self.global_tabu, petri_net.clone(), end))
        else:
            for _ in range(ant_count):
                self.ants.append(Ant(self.map, self.global_tabu, petri_net.clone(), end))
        self.executor_service = ThreadPoolExecutor(max_workers=ant_count)
        self.ant_step_sum = 0
        self.min_prefix = -1
        self.best_ant = None

    def get_ants(self):
        return self.ants

    def get_best_ant(self):
        return self.best_ant

    def get_map(self):
        return self.map

    def get_pheromone_map(self):
        return self.map

    def search(self):
        for idx in range(self.round):
            if self.is_converge():
                break
            futures = self.ants_travel_begin()
            self.dilute_all()
            self.ants_travel_end(futures)
            self.put_pheromone()
            self.i += 1
            if self.on_round is not None:
                self.on_round(idx + 1, self.round, self.min_prefix)
        ant = self.chose_ant()
        if SearchFunctionSwitch.tryBestToFind and ant is None:
            self.i -= 1
            return self.search()
        self.extra_info["antStepSum"] = int(self.ant_step_sum)
        return self.make_result(ant)

    def is_converge(self):
        if self.best_ant is None:
            return False
        if self.ans_is_change:
            self.unchange_count = 0
            self.ans_is_change = False
            return False
        if self.unchange_count > SearchVariable.convergeJudgeCount:
            self.ans_is_change = False
            return True
        self.unchange_count += 1
        self.ans_is_change = False
        return False

    def get_extra_info(self):
        return self.extra_info

    def ants_travel_begin(self):
        futures = []
        for ant in self.ants:
            future = self.executor_service.submit(self._ant_travel, ant)
            futures.append(future)
        return futures

    def _ant_travel(self, ant):
        ant.go_back_home()
        ant.travel()
        return ant

    def ants_travel_end(self, futures):
        for future in futures:
            future.result()
        for ant in self.ants:
            if isinstance(ant, AntWithMemory):
                for marking in ant.get_bad_markings():
                    self.global_tabu.add(marking)
        self.map = self.map_temp
        for ant in self.ants:
            ant.set_map(self.map)
            self.ant_step_sum += ant.get_step()

    def dilute_all(self):
        self.copy()
        for marking in list(self.map_temp.keys()):
            pheromone = self.map_temp[marking]
            pheromone.dilute()

    def put_pheromone(self):
        alive_ant_map = {}
        for ant in self.ants:
            if ant.is_same():
                alive_ant_map[ant] = True
        alive_ants = set(alive_ant_map.keys())
        for ant in self.ants:
            if ant in alive_ants:
                self.ant_alive_plan(ant)
            else:
                self.ant_dead_plan(ant, alive_ants)
        self.renew_best_ant(alive_ants)

    def renew_best_ant(self, alive_ants):
        for ant in alive_ants:
            log = ant.get_log()
            prefix = log[-1].get_curr().get_prefix()
            if self.min_prefix == -1 or self.min_prefix > prefix:
                self.min_prefix = prefix
                self.best_ant = ant.clone()

    def similarity(self, the_ant, other_ant):
        other_ant_path = other_ant.get_tabu()
        same_count = 0
        for log_entity in the_ant.get_log():
            marking = log_entity.get_next()
            if marking in other_ant_path:
                same_count += 1
        return same_count / len(the_ant.get_log()) if the_ant.get_log() else 0

    def find_most_similar_ant(self, ant, alive_ants):
        max_similarity = 0
        the_ant = None
        for alive_ant in alive_ants:
            similarity = self.similarity(ant, alive_ant)
            if max_similarity <= similarity:
                max_similarity = similarity
                the_ant = alive_ant
        return the_ant

    def ant_alive_plan(self, ant):
        if ant is None:
            return
        log = ant.get_log()
        prefix = log[-1].get_curr().get_prefix()
        magnify = self.cacu_magnify(prefix)
        for log_entity in log:
            add_val = SearchVariable.antC // prefix
            add_val *= magnify
            if SearchFunctionSwitch.isSimulatedAnnealing:
                add_val = int(add_val * self.round_rate[self.i])
            marking = log_entity.get_curr()
            tran = log_entity.get_tran()
            pheromone = self.map.get(marking, log_entity.get_pheromone())
            pheromone.add(tran, add_val)
            self.map[marking] = pheromone

    def cacu_magnify(self, prefix):
        if self.min_prefix == -1:
            return 1
        magnify = self.min_prefix - prefix
        if magnify > 0:
            self.ans_is_change = True
        else:
            magnify = 1
        if SearchFunctionSwitch.isSurperAntStrategy:
            return magnify
        return 1

    def ant_dead_plan(self, ant, alive_ants):
        if SearchFunctionSwitch.isRevive:
            if not alive_ants:
                self.ant_alive_plan(self.best_ant)
            else:
                revive_ant = self.find_most_similar_ant(ant, alive_ants)
                self.ant_alive_plan(revive_ant)

    def copy(self):
        self.map_temp = {}
        for marking in self.map:
            self.map_temp[marking] = self.map[marking].clone()

    def chose_ant(self):
        return self.best_ant

    def make_result(self, ant):
        if ant is None:
            return None
        trans = []
        markings = [self.petri_net.get_marking()]
        log = ant.get_log()
        for log_entity in log:
            trans.append(log_entity.get_tran())
            markings.append(log_entity.get_next())
        return Result(trans, markings)


class AntClonyOptimizationWithGreedy(AntClonyOptimization):
    def __init__(self, ant_count, round_count, petri_net, end):
        super().__init__(ant_count, round_count, petri_net, end)
        from .greedy import Greedy
        search = Greedy(petri_net.clone(), end)
        result = search.search()
        petri_net_temp = petri_net.clone()
        for i in range(len(result.get_trans())):
            marking = result.get_markings()[i]
            tran = result.get_trans()[i]
            petri_net_temp.set_marking(marking)
            pheromone = Pheromone(petri_net_temp)
            pheromone.add(tran, 100 * SearchVariable.initPheromone)
            self.map[marking] = pheromone

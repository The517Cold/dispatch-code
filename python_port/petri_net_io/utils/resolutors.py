from .central_container import CentralContainer


class Resolutor:
    def __init__(self):
        self.uuid = None
        self.place_count = 0
        self.tran_count = 0
        self.p_map = {}
        self.t_map = {}
        self.vectors = {}
        self.values = {}
        self.sets = {}
        self.groups = {}
        self.key = None

    def set_uuid(self, uuid):
        self.uuid = uuid

    def set_key(self, key):
        self.key = key

    def deal(self):
        self.place_count = CentralContainer.get("placeCount" + self.uuid)
        self.tran_count = CentralContainer.get("tranCount" + self.uuid)
        self.p_map = CentralContainer.get("pMap" + self.uuid)
        self.t_map = CentralContainer.get("tMap" + self.uuid)
        self.vectors = CentralContainer.get("vectors" + self.uuid)
        self.sets = CentralContainer.get("sets" + self.uuid)
        self.values = CentralContainer.get("values" + self.uuid)
        self.groups = CentralContainer.get("groups" + self.uuid)
        self.pre_resolute()
        self.resolute()
        self.post_resolute()

    def pre_resolute(self):
        raise NotImplementedError()

    def resolute(self):
        raise NotImplementedError()

    def post_resolute(self):
        raise NotImplementedError()


class MapResolutor(Resolutor):
    def __init__(self):
        super().__init__()
        self.map = {}

    def pre_resolute(self):
        self.map = CentralContainer.get_and_delete(self.key + self.uuid)

    def post_resolute(self):
        return


class SetResolutor(Resolutor):
    def __init__(self):
        super().__init__()
        self.set = set()

    def pre_resolute(self):
        self.set = CentralContainer.get_and_delete(self.key + self.uuid)

    def post_resolute(self):
        return


class ValueResolutor(Resolutor):
    def __init__(self):
        super().__init__()
        self.value = 0

    def pre_resolute(self):
        self.value = CentralContainer.get_and_delete(self.key + self.uuid)

    def post_resolute(self):
        return


class CapicityResolutor(MapResolutor):
    def resolute(self):
        capicity = [2 ** 31 - 1] * self.place_count
        for place_name in self.map:
            if place_name not in self.p_map:
                continue
            capicity[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["capicity"] = capicity


class PlaceToPlacesResolutor(MapResolutor):
    def resolute(self):
        place_from_places = [[] for _ in range(self.place_count)]
        for from_place_name in self.map:
            if from_place_name not in self.p_map:
                continue
            from_place_id = self.p_map[from_place_name]
            for to_place_name in self.map[from_place_name].split(" "):
                if to_place_name not in self.p_map:
                    continue
                to_place_id = self.p_map[to_place_name]
                place_from_places[to_place_id].append(from_place_id)
        self.groups["placeFromPlaces"] = place_from_places


class PtimeResolutor(MapResolutor):
    def resolute(self):
        min_delay_p = [0] * self.place_count
        for place_name in self.map:
            if place_name not in self.p_map:
                continue
            min_delay_p[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["minDelayP"] = min_delay_p


class QtimePlacesResolutor(SetResolutor):
    def resolute(self):
        qtime_places = [False] * self.place_count
        for place_name in self.set:
            if place_name not in self.p_map:
                continue
            qtime_places[self.p_map[place_name]] = True
        self.sets["qtimePlaces"] = qtime_places


class QtimeResolutor(ValueResolutor):
    def resolute(self):
        self.values["qtime"] = int(self.value)


class ResidenceTimeResolutor(MapResolutor):
    def resolute(self):
        max_residence_time = [2 ** 31 - 1] * self.place_count
        for place_name in self.map:
            if place_name not in self.p_map:
                continue
            max_residence_time[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["maxResidenceTime"] = max_residence_time


class ResourcePlaceResolutor(SetResolutor):
    def resolute(self):
        is_resource = [False] * self.place_count
        for place_name in self.set:
            if place_name not in self.p_map:
                continue
            is_resource[self.p_map[place_name]] = True
        self.sets["isResource"] = is_resource


class TtimeResolutor(MapResolutor):
    def resolute(self):
        min_delay_t = [0] * self.tran_count
        for tran_name in self.map:
            if tran_name not in self.t_map:
                continue
            min_delay_t[self.t_map[tran_name]] = int(self.map[tran_name])
        self.vectors["minDelayT"] = min_delay_t


class MovePlacesResolutor(SetResolutor):
    """解析 movePlaces 字段，标记参与超时公式中需扣除延迟的"移动库所"。

    movePlaces 中的库所是本次变迁前置库所的子集；在计算 qtime 超时时，
    这些库所的 ptime 延迟视为合法处理时间，从总耗时中扣除。
    """

    def resolute(self):
        move_places = [False] * self.place_count
        for place_name in self.set:
            if place_name not in self.p_map:
                continue
            move_places[self.p_map[place_name]] = True
        self.sets["movePlaces"] = move_places


class CompleteTimeResolutor(SetResolutor):
    """解析 completeTime 字段，表示工件完成加工的固定时间常量。

    文件中使用 "completeTime:2" 冒号语法，解析器将其归入 set_info（单元素集合）。
    本 resolutor 从集合中取出该单一整数值，存入 values["completeTime"]。

    该常量在 qtime 超时判断公式中从总耗时中扣除，以剔除必要的
    装卸/完工时间对超时判断的影响。
    """

    def resolute(self):
        value = int(next(iter(self.set))) if self.set else 0
        self.values["completeTime"] = value

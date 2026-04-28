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
            capicity[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["capicity"] = capicity


class PlaceToPlacesResolutor(MapResolutor):
    def resolute(self):
        place_from_places = [[] for _ in range(self.place_count)]
        for from_place_name in self.map:
            from_place_id = self.p_map[from_place_name]
            for to_place_name in self.map[from_place_name].split(" "):
                to_place_id = self.p_map[to_place_name]
                place_from_places[to_place_id].append(from_place_id)
        self.groups["placeFromPlaces"] = place_from_places


class PtimeResolutor(MapResolutor):
    def resolute(self):
        min_delay_p = [0] * self.place_count
        for place_name in self.map:
            min_delay_p[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["minDelayP"] = min_delay_p


class QtimePlacesResolutor(SetResolutor):
    def resolute(self):
        qtime_places = [False] * self.place_count
        for place_name in self.set:
            qtime_places[self.p_map[place_name]] = True
        self.sets["qtimePlaces"] = qtime_places


class QtimeResolutor(ValueResolutor):
    def resolute(self):
        self.values["qtime"] = int(self.value)


class ResidenceTimeResolutor(MapResolutor):
    def resolute(self):
        max_residence_time = [2 ** 31 - 1] * self.place_count
        for place_name in self.map:
            max_residence_time[self.p_map[place_name]] = int(self.map[place_name])
        self.vectors["maxResidenceTime"] = max_residence_time


class ResourcePlace(SetResolutor):
    def resolute(self):
        is_resource = [False] * self.place_count
        for place_name in self.set:
            is_resource[self.p_map[place_name]] = True
        self.sets["isResource"] = is_resource


class TtimeResolutor(MapResolutor):
    def resolute(self):
        min_delay_t = [0] * self.tran_count
        for tran_name in self.map:
            min_delay_t[self.t_map[tran_name]] = int(self.map[tran_name])
        self.vectors["minDelayT"] = min_delay_t

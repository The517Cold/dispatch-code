import uuid as uuid_lib
from ..input_entity import PetriNetFile
from .central_container import CentralContainer
from . import resolutors


class MatrixTranslator:
    def __init__(self, petri_net_file):
        self.petri_net_file = petri_net_file
        self.a_matrix = []
        self.pre = []
        self.post = []
        self.vectors = {}
        self.sets = {}
        self.groups = {}
        self.values = {}
        self.p_map = {}
        self.t_map = {}
        self.p_map_v = {}
        self.t_map_v = {}

    def translate(self):
        self.pre_custom_strategy()
        self._maintain_strategy()
        self.post_custom_strategy()

    def _make_matrix(self):
        count_p = 0
        count_t = 0
        for place_entity in self.petri_net_file.net_struct:
            self.p_map[place_entity.place_name] = count_p
            count_p += 1
            for arc_entity in place_entity.pre:
                if arc_entity.tran_name not in self.t_map:
                    self.t_map[arc_entity.tran_name] = count_t
                    count_t += 1
            for arc_entity in place_entity.post:
                if arc_entity.tran_name not in self.t_map:
                    self.t_map[arc_entity.tran_name] = count_t
                    count_t += 1
        for key, value in self.t_map.items():
            self.t_map_v[value] = key
        for key, value in self.p_map.items():
            self.p_map_v[value] = key
        self.pre = [[0 for _ in range(len(self.t_map))] for _ in range(len(self.p_map))]
        self.post = [[0 for _ in range(len(self.t_map))] for _ in range(len(self.p_map))]
        for place_entity in self.petri_net_file.net_struct:
            place_name = place_entity.place_name
            for arc_entity in place_entity.pre:
                tran_name = arc_entity.tran_name
                self.post[self.p_map[place_name]][self.t_map[tran_name]] = arc_entity.weight
            for arc_entity in place_entity.post:
                tran_name = arc_entity.tran_name
                self.pre[self.p_map[place_name]][self.t_map[tran_name]] = arc_entity.weight
        self.a_matrix = [[0 for _ in range(len(self.t_map))] for _ in range(len(self.p_map))]
        for i in range(len(self.a_matrix)):
            for j in range(len(self.a_matrix[0])):
                self.a_matrix[i][j] = self.post[i][j] - self.pre[i][j]

    def _make_start_and_end(self):
        place_count = len(self.a_matrix)
        p_info = [0] * place_count
        end = [-1] * place_count
        start_marking = self.petri_net_file.map_info.get("startMarking", {})
        goal_marking = self.petri_net_file.map_info.get("goalPlace", {})
        if "startMarking" in self.petri_net_file.map_info:
            del self.petri_net_file.map_info["startMarking"]
        if "goalPlace" in self.petri_net_file.map_info:
            del self.petri_net_file.map_info["goalPlace"]
        for place in start_marking:
            p_info[self.p_map[place]] = int(str(start_marking[place]))
        for place in goal_marking:
            end[self.p_map[place]] = int(goal_marking[place])
        self.vectors["pInfo"] = p_info
        self.vectors["end"] = end

    def pre_custom_strategy(self):
        raise NotImplementedError()

    def _maintain_strategy(self):
        self._make_matrix()
        self._make_start_and_end()

    def post_custom_strategy(self):
        raise NotImplementedError()


class CustomMatrixTranslator(MatrixTranslator):
    def pre_custom_strategy(self):
        return

    def post_custom_strategy(self):
        uuid = str(uuid_lib.uuid4())
        self._put_info_into_central_container(uuid)
        for key in list(self.petri_net_file.map_info.keys()):
            resolutor = self._get_resolutor(key, uuid)
            CentralContainer.put(key + uuid, self.petri_net_file.map_info.get(key))
            resolutor.deal()
        for key in list(self.petri_net_file.set_info.keys()):
            resolutor = self._get_resolutor(key, uuid)
            CentralContainer.put(key + uuid, self.petri_net_file.set_info.get(key))
            resolutor.deal()
        for key in list(self.petri_net_file.value_info.keys()):
            resolutor = self._get_resolutor(key, uuid)
            CentralContainer.put(key + uuid, self.petri_net_file.value_info.get(key))
            resolutor.deal()
        self._remove_info_from_central_container(uuid)

    def _get_resolutor(self, key, uuid):
        class_name = self._capture_name(key) + "Resolutor"
        clazz = getattr(resolutors, class_name, None)
        if clazz is None:
            raise RuntimeError("resolver没找到")
        resolutor = clazz()
        resolutor.set_uuid(uuid)
        resolutor.set_key(key)
        return resolutor

    def _put_info_into_central_container(self, uuid):
        place_count = len(self.a_matrix)
        tran_count = len(self.a_matrix[0])
        CentralContainer.put("placeCount" + uuid, place_count)
        CentralContainer.put("tranCount" + uuid, tran_count)
        CentralContainer.put("pMap" + uuid, self.p_map)
        CentralContainer.put("tMap" + uuid, self.t_map)
        CentralContainer.put("vectors" + uuid, self.vectors)
        CentralContainer.put("sets" + uuid, self.sets)
        CentralContainer.put("groups" + uuid, self.groups)
        CentralContainer.put("values" + uuid, self.values)

    def _remove_info_from_central_container(self, uuid):
        CentralContainer.get_and_delete("placeCount" + uuid)
        CentralContainer.get_and_delete("tranCount" + uuid)
        CentralContainer.get_and_delete("pMap" + uuid)
        CentralContainer.get_and_delete("tMap" + uuid)
        CentralContainer.get_and_delete("vectors" + uuid)
        CentralContainer.get_and_delete("sets" + uuid)
        CentralContainer.get_and_delete("groups" + uuid)
        CentralContainer.get_and_delete("values" + uuid)

    def _capture_name(self, name):
        return name[:1].upper() + name[1:]


class NormalPetriNetTranslator(MatrixTranslator):
    def pre_custom_strategy(self):
        return

    def post_custom_strategy(self):
        return

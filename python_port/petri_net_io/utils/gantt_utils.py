from ..output_entity import GanttData
from ...petri_net_platform.marking import TTPPNMarkingByTokenWithResTime


class SplitTool:
    def __init__(self, gantt_tool):
        self.gantt_tool = gantt_tool

    def split_tran(self, gantt_data_list, index, tran):
        t_after_p = self.gantt_tool.get_t_after_p()
        lst = t_after_p.get(tran.number)
        id_list = []
        for place in lst:
            m = self.gantt_tool.get_id(index, place, False)
            if m is not None:
                id_list.append(m)
        if not id_list:
            id_list.append(-1)
        time = self.gantt_tool.get_prefix_list()[index + 1]
        tran_index = self.gantt_tool.get_t_key_index()[tran.number]
        tran_time = self.gantt_tool.get_t_time()[tran_index]
        gantt_data_list.append(GanttData(tran.number + "(T)", time - tran_time, time, id_list))

    def split_after_tran(self, gantt_data_list, index, tran, place):
        tran_index = self.gantt_tool.get_t_key_index()[tran.number]
        tran_time = self.gantt_tool.get_t_time()[tran_index]
        time = self.gantt_tool.get_prefix_list()[index + 1]
        id_list = []
        token_id = self.gantt_tool.get_id(index, place.number, True)
        if token_id is None:
            token_id = self.gantt_tool.get_id(index, place.number, False)
        id_list.append(token_id)
        gantt_data_list.append(GanttData(tran.number + "(T1)", time - tran_time - place.separate[0], time - tran_time, id_list))
        gantt_data_list.append(GanttData(tran.number + "(T3)", time, time + place.separate[1], id_list))

    def split_before_tran(self, gantt_data_list, index, tran, place):
        time = self.gantt_tool.get_prefix_list()[index + 1]
        tran_index = self.gantt_tool.get_t_key_index()[tran.number]
        tran_time = self.gantt_tool.get_t_time()[tran_index]
        place_index = self.gantt_tool.get_p_key_index()[place.number]
        place_time = self.gantt_tool.get_p_time()[place_index] - place.separate[0] - place.separate[1]
        id_list = []
        token_id = self.gantt_tool.get_id(index, place.number, False)
        id_list.append(token_id)
        gantt_data_list.append(GanttData(tran.number + "(T1)", time - tran_time - place.separate[0] - place.separate[1], time - tran_time - place.separate[1], id_list))
        gantt_data_list.append(GanttData(tran.number + "(T3)", time - place.separate[1], time, id_list))
        gantt_data_list.append(GanttData(tran.number + "(T2)", time, time + place_time, id_list))


class GanttTool:
    def __init__(self, result, petri_net_file, matrix_translator, need_move_translator):
        self.split_tool = SplitTool(self)
        self.color_map = need_move_translator.color_map
        self.input_map = need_move_translator.input_map
        self.p_time = matrix_translator.vectors.get("minDelayP")
        self.t_time = matrix_translator.vectors.get("minDelayT")
        self.t_key_index = matrix_translator.t_map
        self.p_key_index = matrix_translator.p_map
        self.key_p_map = matrix_translator.p_map_v
        self.key_t_map = matrix_translator.t_map_v
        self.tran_list = []
        self.move_name_list = list(need_move_translator.move_name_list)
        for tran in result.get_trans():
            self.tran_list.append(int(matrix_translator.t_map_v[tran]))
        self.prefix_list = []
        self.marking_list = []
        for marking in result.get_markings():
            self.prefix_list.append(marking.get_prefix())
            if isinstance(marking, TTPPNMarkingByTokenWithResTime):
                self.marking_list.append(marking)
        self.p_before_t = {}
        self.p_after_t = {}
        for place_entity in petri_net_file.net_struct:
            list_before = []
            list_after = []
            place_name = place_entity.place_name
            for arc_entity in place_entity.pre:
                list_before.append(arc_entity.tran_name)
            self.p_before_t[place_name] = list_before
            for arc_entity in place_entity.post:
                list_after.append(arc_entity.tran_name)
            self.p_after_t[place_name] = list_after
        matrix = matrix_translator.a_matrix
        pre = matrix_translator.pre
        post = matrix_translator.post
        self.t_after_p = {}
        self.t_before_p = {}
        for i in range(len(matrix[0])):
            after_places = []
            before_places = []
            for j in range(len(matrix)):
                if post[j][i] > 0:
                    after_places.append(self.key_p_map[j])
                if pre[j][i] > 0:
                    before_places.append(self.key_p_map[j])
            tran = self.key_t_map[i]
            self.t_after_p[tran] = after_places
            self.t_before_p[tran] = before_places

    def get_p_time(self):
        return self.p_time

    def get_t_time(self):
        return self.t_time

    def get_t_key_index(self):
        return self.t_key_index

    def get_p_key_index(self):
        return self.p_key_index

    def get_key_t_map(self):
        return self.key_t_map

    def get_key_p_map(self):
        return self.key_p_map

    def get_prefix_list(self):
        return self.prefix_list

    def get_tran_list(self):
        return self.tran_list

    def get_color_map(self):
        return self.color_map

    def get_input_map(self):
        return self.input_map

    def get_move_name_list(self):
        return self.move_name_list

    def get_t_after_p(self):
        return self.t_after_p

    def get_t_before_p(self):
        return self.t_before_p

    def get_p_before_t(self):
        return self.p_before_t

    def get_p_after_t(self):
        return self.p_after_t

    def get_tran_index(self, tran_move):
        index_map = {}
        tran_list = self.tran_list
        for i in range(len(tran_list)):
            for move in tran_move:
                if move.number == str(tran_list[i]):
                    index_map[i] = move
        return index_map

    def add_color(self, gantt_data_list):
        for gantt_data in gantt_data_list:
            for lst in self.color_map:
                if gantt_data.id in lst:
                    gantt_data.color = self.color_map[lst]

    def add_move_name(self, gantt_data_list, name):
        for gantt_data in gantt_data_list:
            gantt_data.set_move_name(name)

    def add_place_by_after_tran(self, gantt_data_list, tran_index_map, place_move):
        for index in tran_index_map:
            tran = tran_index_map[index]
            place = None
            string_list = self.t_before_p.get(tran.number)
            for s in string_list:
                flag = False
                for move in place_move:
                    if move.number == s:
                        place = move
                        flag = True
                        break
                if flag:
                    break
            if place is not None:
                self.split_tool.split_after_tran(gantt_data_list, index, tran, place)

    def add_tran(self, gantt_data_list, tran_index_map):
        for index in tran_index_map:
            tran = tran_index_map[index]
            self.split_tool.split_tran(gantt_data_list, index, tran)

    def add_place_by_before_tran(self, gantt_data_list, tran_index_map, place_move):
        for index in tran_index_map:
            tran = tran_index_map[index]
            place = None
            string_list = self.t_after_p.get(tran.number)
            for s in string_list:
                flag = False
                for move in place_move:
                    if move.number == s:
                        place = move
                        flag = True
                        break
                if flag:
                    break
            if place is not None:
                self.split_tool.split_before_tran(gantt_data_list, index, tran, place)

    def get_id(self, tran_index, place, is_before):
        first_marking = self.marking_list[0]
        list_tokens = list(first_marking.t_info[1])
        min_id = list_tokens[0].get_id()
        max_id = list_tokens[-1].get_id()
        after_marking = self.marking_list[tran_index + 1]
        before_marking = self.marking_list[tran_index]
        place_index = self.p_key_index[place]
        before_token_list = list(before_marking.t_info[place_index])
        after_token_list = list(after_marking.t_info[place_index])
        if is_before:
            for token in before_token_list:
                flag = False
                for token1 in after_token_list:
                    if token.get_id() == token1.get_id():
                        flag = True
                        break
                if not flag and max_id >= token.get_id() >= min_id:
                    return token.get_id()
        else:
            for token in after_token_list:
                flag = False
                for token1 in before_token_list:
                    if token.get_id() == token1.get_id():
                        flag = True
                        break
                if not flag and max_id >= token.get_id() >= min_id:
                    return token.get_id()
        return None

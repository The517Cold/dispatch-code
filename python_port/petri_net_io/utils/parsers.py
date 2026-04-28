from ..output_entity import GanttData
from .gantt_utils import SplitTool


class Parser:
    def __init__(self):
        self.gantt_tool = None

    def set_gantt_tool(self, gantt_tool):
        self.gantt_tool = gantt_tool

    def parse(self, move_name):
        raise NotImplementedError()


class COParser(Parser):
    def parse(self, move_name):
        split_tool = SplitTool(self.gantt_tool)
        lst = []
        gantt_data_list1 = []
        gantt_data_list2 = []
        gantt_data_list3 = []
        move_list = self.gantt_tool.get_input_map().get(move_name)
        place_move = move_list[0]
        before_tran_move = move_list[1]
        after_tran_move = move_list[2]
        place = place_move[0]
        before_index_map = self.gantt_tool.get_tran_index(before_tran_move)
        after_index_map = self.gantt_tool.get_tran_index(after_tran_move)
        count = 1
        for index in before_index_map:
            tran = before_index_map[index]
            if count % 3 == 0:
                split_tool.split_before_tran(gantt_data_list1, index, tran, place)
            elif count % 3 == 1:
                split_tool.split_before_tran(gantt_data_list2, index, tran, place)
            else:
                split_tool.split_before_tran(gantt_data_list3, index, tran, place)
            count += 1
        for index in after_index_map:
            tran = after_index_map[index]
            if count % 3 == 0:
                split_tool.split_after_tran(gantt_data_list1, index, tran, place)
            elif count % 3 == 1:
                split_tool.split_after_tran(gantt_data_list2, index, tran, place)
            else:
                split_tool.split_after_tran(gantt_data_list3, index, tran, place)
            count += 1
        lst.append(gantt_data_list1)
        lst.append(gantt_data_list2)
        lst.append(gantt_data_list3)
        for list_item in lst:
            list_item.sort()
            self.gantt_tool.add_color(list_item)
            self.gantt_tool.add_move_name(list_item, move_name)
        return lst


class LLParser(Parser):
    def parse(self, move_name):
        lst = []
        gantt_data_list = []
        move_list = self.gantt_tool.get_input_map().get(move_name)
        place_move = move_list[0]
        f_time = self.gantt_tool.get_p_time()[self.gantt_tool.get_p_key_index()[place_move[0].number]]
        c_time = self.gantt_tool.get_p_time()[self.gantt_tool.get_p_key_index()[place_move[1].number]]
        moves_a = move_list[1]
        moves_b = move_list[2]
        l_tran_list = []
        l_tran_index = []
        for i in range(len(self.gantt_tool.get_tran_list())):
            for move in moves_a:
                if move.number == str(self.gantt_tool.get_tran_list()[i]):
                    l_tran_index.append(i)
                    l_tran_list.append(move)
            for move in moves_b:
                if move.number == str(self.gantt_tool.get_tran_list()[i]):
                    l_tran_index.append(i)
                    l_tran_list.append(move)
        for i in range(len(l_tran_list)):
            index = l_tran_index[i]
            time = self.gantt_tool.get_prefix_list()[index + 1]
            id_list = []
            if l_tran_list[i] in moves_b:
                after_place_list = self.gantt_tool.get_t_after_p().get(l_tran_list[i].number)
                for place in after_place_list:
                    m = self.gantt_tool.get_id(index, place, False)
                    if m is not None:
                        id_list.append(m)
            if l_tran_list[i] in moves_a:
                before_place_list = self.gantt_tool.get_t_before_p().get(l_tran_list[i].number)
                for place in before_place_list:
                    m = self.gantt_tool.get_id(index, place, True)
                    if m is not None:
                        id_list.append(m)
            if not id_list:
                id_list.append(-1)
            tran_time = self.gantt_tool.get_t_time()[self.gantt_tool.get_t_key_index()[l_tran_list[i].number]]
            gantt_data_list.append(GanttData(l_tran_list[i].number + "(T1)", time - tran_time - l_tran_list[i].separate[0], time - tran_time, id_list))
            gantt_data_list.append(GanttData(l_tran_list[i].number + "(T3)", time, time + l_tran_list[i].separate[1], id_list))
            if i == len(l_tran_list) - 1:
                break
            if l_tran_list[i] in moves_a:
                if l_tran_list[i + 1] in moves_b:
                    start = time + l_tran_list[i].separate[1]
                    end = start + f_time - (l_tran_list[i].separate[0] * 2)
                    gantt_data_list.append(GanttData(l_tran_list[i].number + "(T2)", start, end, id_list))
            if l_tran_list[i] in moves_b:
                if l_tran_list[i + 1] in moves_a:
                    start = time + l_tran_list[i].separate[1]
                    end = start + c_time - (l_tran_list[i].separate[0] * 2)
                    gantt_data_list.append(GanttData(l_tran_list[i].number + "(T2)", start, end, id_list))
        n = len(move_name) - 2
        self.gantt_tool.add_move_name(gantt_data_list, move_name[len(move_name) - n :])
        self.gantt_tool.add_color(gantt_data_list)
        gantt_data_list.sort()
        lst.append(gantt_data_list)
        return lst


class LPParser(Parser):
    def parse(self, move_name):
        queue_list = []
        gantt_data_list = []
        move_list = self.gantt_tool.get_input_map().get(move_name)
        place_move = move_list[0]
        tran_move = move_list[1]
        start_index_map = self.gantt_tool.get_tran_index(tran_move)
        self.gantt_tool.add_place_by_after_tran(gantt_data_list, start_index_map, place_move)
        gantt_data_list.sort()
        self.gantt_tool.add_move_name(gantt_data_list, move_name)
        self.gantt_tool.add_color(gantt_data_list)
        queue_list.append(gantt_data_list)
        return queue_list


class PMParser(Parser):
    def parse(self, move_name):
        lst = []
        gantt_data_list = []
        move_list = self.gantt_tool.get_input_map().get(move_name)
        if len(move_list) > 4:
            clean_move = move_list[3]
            start_index_map = self.gantt_tool.get_tran_index(clean_move)
            self.gantt_tool.add_tran(gantt_data_list, start_index_map)
        place_move = move_list[0]
        before_tran_move = move_list[1]
        after_tran_move = move_list[2]
        before_index_map = self.gantt_tool.get_tran_index(before_tran_move)
        self.gantt_tool.add_place_by_before_tran(gantt_data_list, before_index_map, place_move)
        after_index_map = self.gantt_tool.get_tran_index(after_tran_move)
        self.gantt_tool.add_place_by_after_tran(gantt_data_list, after_index_map, place_move)
        gantt_data_list.sort()
        self.gantt_tool.add_move_name(gantt_data_list, move_name)
        self.gantt_tool.add_color(gantt_data_list)
        lst.append(gantt_data_list)
        return lst


class TTParser(Parser):
    def parse(self, move_name):
        queue_list = []
        gantt_data_list = []
        move_list = self.gantt_tool.get_input_map().get(move_name)
        tran_move = move_list[1]
        start_index_map = self.gantt_tool.get_tran_index(tran_move)
        self.gantt_tool.add_tran(gantt_data_list, start_index_map)
        gantt_data_list.sort()
        n = len(move_name) - 2
        self.gantt_tool.add_move_name(gantt_data_list, move_name[len(move_name) - n :])
        self.gantt_tool.add_color(gantt_data_list)
        queue_list.append(gantt_data_list)
        return queue_list

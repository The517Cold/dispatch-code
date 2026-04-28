from .parsers import COParser, LLParser, LPParser, PMParser, TTParser


class GanttTranslator:
    def __init__(self, gantt_tool):
        self.gantt_data_queue = {}
        self.gantt_tool = gantt_tool

    def translate(self):
        move_name_list = list(self.gantt_tool.get_move_name_list())
        for s in move_name_list:
            name = self._capture_name(s)
            parser = self._create_parser(name)
            if parser is None:
                raise RuntimeError("resolver没找到")
            parser.set_gantt_tool(self.gantt_tool)
            queue_list = parser.parse(s)
            if len(queue_list) > 1:
                k = 1
                for queue in queue_list:
                    self.gantt_data_queue[s + str(k)] = queue
                    self.gantt_tool.get_move_name_list().append(s + str(k))
                    k += 1
            else:
                self.gantt_data_queue[s] = queue_list[0]

    def _create_parser(self, name):
        prefix = name[:2]
        if prefix == "CO":
            return COParser()
        if prefix == "LL":
            return LLParser()
        if prefix == "LP":
            return LPParser()
        if prefix == "PM":
            return PMParser()
        if prefix == "TT":
            return TTParser()
        return None

    def _capture_name(self, name):
        return name[:1].upper() + name[1:]

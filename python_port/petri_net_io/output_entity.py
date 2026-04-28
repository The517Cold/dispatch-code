class GanttData:
    def __init__(self, item_id, start, end, token_id=None):
        self.id = item_id
        self.start = start
        self.end = end
        self.token_id = token_id
        self.color = None
        self.move_name = None

    def get_move_name(self):
        return self.move_name

    def set_move_name(self, move_name):
        self.move_name = move_name

    def __lt__(self, other):
        return self.start < other.start


class Move:
    def __init__(self, number=None, separate=None):
        self.number = number
        self.separate = separate


class MoveEntity:
    def __init__(self):
        self.module_name = None
        self.move_type = 0
        self.mat_id = []
        self.src = None
        self.src_slot = []
        self.dest = None
        self.dest_slot = []
        self.robot_slot = []
        self.start_time = 0.0
        self.end_time = 0.0

class ArcEntity:
    def __init__(self, tran_name=None, weight=0):
        self.tran_name = tran_name
        self.weight = weight


class PlaceEntity:
    def __init__(self, place_name=None, pre=None, post=None):
        self.place_name = place_name
        self.pre = pre if pre is not None else []
        self.post = post if post is not None else []


class PetriNetFile:
    def __init__(self):
        self.map_info = {}
        self.set_info = {}
        self.value_info = {}
        self.net_struct = []
        self.EFline = None

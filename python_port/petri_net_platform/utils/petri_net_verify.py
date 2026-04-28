class PetriNetVerify:
    def __init__(self, petri_net):
        self.petri_net = petri_net
        self.tabu = set()
        self.enable_prefix_trans = []

    def get_enable_prefix_trans(self):
        return self.enable_prefix_trans

    def verify_single_tran(self, tran):
        if not self.petri_net.enable(tran):
            return None
        return self.petri_net.launch(tran)

    def verify_trans(self, trans):
        curr = self.petri_net.clone()
        self.enable_prefix_trans = []
        for tran in trans:
            if not curr.enable(tran):
                return None
            self.enable_prefix_trans.append(tran)
            curr.set_marking(curr.launch(tran))
        return curr.get_marking()

    def is_dead(self, trans):
        curr = self.petri_net.clone()
        for tran in trans:
            if not curr.enable(tran):
                return True
            marking = curr.launch(tran)
            self.tabu.add(marking)
            curr.set_marking(marking)
        for tran in range(curr.get_trans_count()):
            if curr.enable(tran) and curr.launch(tran) not in self.tabu:
                return True
        return False

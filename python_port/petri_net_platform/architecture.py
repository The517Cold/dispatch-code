class Marking:
    def get_prefix(self):
        raise NotImplementedError()

    def get_p_info(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()


class PetriNet:
    def launch(self, tran):
        raise NotImplementedError()

    def enable(self, tran):
        raise NotImplementedError()

    def get_marking(self):
        raise NotImplementedError()

    def set_marking(self, marking):
        raise NotImplementedError()

    def get_trans_count(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()


class Search:
    def search(self):
        raise NotImplementedError()

    def get_extra_info(self):
        raise NotImplementedError()


class HasResideceTime(Marking):
    def is_over_residece_time(self):
        raise NotImplementedError()

    def get_residence_time(self, place, token=None):
        raise NotImplementedError()

    def get_over_residence_time_place(self):
        raise NotImplementedError()

    def get_time(self, place):
        raise NotImplementedError()

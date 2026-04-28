class TabuCondition:
    def __init__(self):
        self.tabu_condition = {}
        self.is_over_residece_time = False

    def set_over_residece_time(self, over_residece_time):
        self.is_over_residece_time = over_residece_time

    def judge(self, marking):
        if not self.is_over_residece_time:
            return False
        for place in self.tabu_condition:
            for token in self.tabu_condition[place]:
                time_val = self.tabu_condition[place][token]
                if time_val <= marking.get_residence_time(place, token):
                    return False
        return True

    def renew(self, place, token, time_val):
        place_condition = self.tabu_condition.get(place, {})
        old_time = place_condition.get(token, 2 ** 31 - 1)
        if time_val != -1 and time_val < old_time:
            place_condition[token] = time_val
        self.tabu_condition[place] = place_condition

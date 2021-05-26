"""
File a scheduler used to change hyper-parameter values during training.
"""


class HyperParamScheduler:
    """
    Updates hyper-parameters during the training session.
    """

    def __init__(self, min_v, max_v, increase_decrease):
        """
        Two modes supported:
            1. Value starts at min_v and increases to max_v by 0.5 num_batches, and then returns to min_v (when
            increase_decrease == True)
            2. Value starts at max_v and decreases to min_v by 0.5 num_batches, and then returns to max_v (when
            increase_decrease == False)
        """

        self.min = min_v
        self.max = max_v
        self.increase_decrease = increase_decrease

        self.num_batches = None
        self.slope = None

        if self.increase_decrease:
            self.curr_value = self.min
        else:
            self.curr_value = self.max

    def __call__(self, batch_i):
        if batch_i < (self.num_batches / 2):
            if self.increase_decrease:
                self.curr_value += self.slope
            else:
                self.curr_value -= self.slope
        else:
            if self.increase_decrease:
                self.curr_value -= self.slope
            else:
                self.curr_value += self.slope

        return self.curr_value

    def set_slope(self, num_batches):
        self.num_batches = num_batches
        self.slope = (self.max - self.min) / (num_batches / 2)

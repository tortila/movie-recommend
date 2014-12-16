import numpy as np

class Predictor:

    def __init__(self, parsed_data, users, movies):
        self.matrix = np.zeros((users, movies))
        self.construct_r(parsed_data)
        self.r_avg = self.get_global_avg()
        print self.r_avg

    def construct_r(self, raw_data):
        for row in raw_data:
            self.matrix[row[0] - 1, row[1] - 1] = row[2]

    def get_global_avg(self):
        return sum(i for row in self.matrix for i in row if i) * 1.0 / np.count_nonzero(self.matrix)

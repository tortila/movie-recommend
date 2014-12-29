import numpy as np
import math as m
import matplotlib.pyplot as plt

TRAINING_FRACTION = 0.05
IMPROVED = "improved"
BASELINE = "baseline"

class Predictor:

    def __init__(self, mode, training_data, test_data):
        self.mode = mode
        if self.mode == BASELINE:
            self.init_baseline(training_data, test_data)
        elif self.mode == IMPROVED:
            self.init_improved(training_data, test_data)

    def init_baseline(self, training_data, test_data):
        self.r_avg = 0.0
        self.ratings_number = np.count_nonzero(training_data)
        self.training_size = self.get_training_size()
        self.training_indices = np.transpose(np.nonzero(training_data))
        self.matrix_A, self.vector_y = self.construct_A(training_data, training_data.shape[0], training_data.shape[1])
        self.bias = self.get_bias(self.matrix_A, self.vector_y)[0]
        self.baseline_matrix = self.get_baseline_matrix(training_data)
        self.rmse_training = self.get_rmse_training(training_data)
        self.rmse_test = self.get_rmse_test(test_data)

    def init_improved(self, training_data, test_data):
        self.init_baseline(training_data, test_data)

    # construct matrix A (N rows, M columns) where N is the number of training data points and M is number of (users + movies)
    def construct_A(self, matrix_R, users, movies):
        # select training set of matrix R
        np.random.shuffle(self.training_indices)
        self.training_indices = self.training_indices[:self.training_size]
        # calculate average rating of training set
        sum = 0
        for row in self.training_indices:
            sum += matrix_R[row[0], row[1]]
        self.r_avg = sum * 1.0 / self.training_size
        # construct the matrix A
        A = np.zeros((self.training_size, users + movies))
        r = np.zeros(self.training_size)
        i = 0
        for row in self.training_indices:
            A[i, int(row[0])] = 1
            A[i, int(row[1]) + users] = 1
            r[i] = matrix_R[row[0], row[1]] - self.r_avg
            i += 1
        return A, r

    def get_bias(self, A, y):
        # rcond to avoid numerical errors
        return np.linalg.lstsq(A, y, rcond=1e-3)
    
    def get_training_size(self):
        return int(self.ratings_number * TRAINING_FRACTION)

    def get_baseline_matrix(self, umr):
        users = umr.shape[0]
        movies = umr.shape[1]
        r_baseline = np.zeros((users, movies))
        for user in range(users):
            for movie in range(movies):
                r_sum = self.r_avg + self.bias[user] + self.bias[user + movie]
                if r_sum < 1.0:
                    r_sum = 1.0
                if r_sum > 5.0:
                    r_sum = 5.0
                r_baseline[user, movie] = r_sum
        # round to the nearest integer
        return r_baseline

    def get_rmse_training(self, training_set):
        training_sum = 0.0
        users = training_set.shape[0]
        movies = training_set.shape[1]
        for user in range(users):
            for movie in range(movies):
                if training_set[user, movie] != 0:
                    training_sum += (np.rint(self.baseline_matrix[user, movie]) - training_set[user, movie]) ** 2
        training = m.sqrt(1.0 / self.ratings_number * training_sum)
        return np.around(training, decimals = 3)

    def get_rmse_test(self, test_set):
        test_sum = 0.0
        source = self.baseline_matrix
        if self.mode == IMPROVED:
            pass
            # source = improved predictor's matrix
        for rating in test_set:
            test_sum += (np.rint(source[rating[0] - 1, rating[1] - 1]) - rating[2]) ** 2
        test = m.sqrt(1.0 / len(test_set) * test_sum)
        return np.around(test, decimals = 3)

    # to be called from main program
    def calculate_absolute_errors(self, test_set):
        source = self.baseline_matrix
        filename = "abs_errors_" + self.mode + ".png"
        if self.mode == IMPROVED:
            pass
            # source = improved predictor's matrix
        # plot a histogram
        hist_data = [(abs(test_set[i][2] - source[test_set[i][0] - 1, test_set[i][1] - 1])) for i in range(len(test_set))]
        hist, bins = np.histogram(hist_data, bins = range(10))
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align = "center", width = 0.7)
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.title("Histogram of the distribution of the absolute errors for " + self.mode + " predictor")
        plt.grid(True)
        plt.savefig(filename)
        return [x for x in hist if x > 0]

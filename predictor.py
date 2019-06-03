import numpy as np
import math as m
import matplotlib

# a workaround to avoid depending on _tkinter package (the default "tk" backend is not used anyway)
from parser import Mode

matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

TRAINING_FRACTION = 0.5
IMPROVED = "improved"
BASELINE = "baseline"
NONE = 100.0  # to distinguish unavailable data
NEIGHBOURS_NUMBER = 20


class Predictor:
    def __init__(self, mode: Mode, training_data, test_data):
        self.mode = mode
        if self.mode == Mode.BASELINE:
            self.init_baseline(training_data, test_data)
        elif self.mode == Mode.IMPROVED:
            self.init_improved(training_data, test_data)

    def init_baseline(self, training_data, test_data):
        self.r_avg = 0.0
        self.users = training_data.shape[0]
        self.movies = training_data.shape[1]
        self.ratings_number = np.count_nonzero(training_data)
        self.training_size = self.get_training_size()
        self.training_indices = np.transpose(np.nonzero(training_data))
        self.matrix_A, self.vector_y = self.construct_a_matrix(training_data)
        self.bias = self.get_bias(self.matrix_A, self.vector_y)[0]
        self.baseline_matrix = self.get_baseline_matrix()
        self.rmse_training = self.get_rmse_training(training_data)
        self.rmse_test = self.get_rmse_test(test_data, self.baseline_matrix)

    def init_improved(self, training_data, test_data):
        self.init_baseline(training_data, test_data)
        self.difference_matrix = self.get_difference_matrix(training_data)
        self.distance_matrix = self.calculate_distance_matrix()
        self.improved_matrix = self.get_improved_matrix(training_data)
        self.rmse_test_improved = self.get_rmse_test(test_data, self.improved_matrix)

    def construct_a_matrix(self, matrix_R):
        # select training set of matrix R
        np.random.shuffle(self.training_indices)
        self.training_indices = self.training_indices[: self.training_size]
        # calculate average rating of training set
        sum = 0
        for row in self.training_indices:
            sum += matrix_R[row[0], row[1]]
        self.r_avg = sum * 1.0 / self.training_size
        # construct the matrix A
        A = np.zeros((self.training_size, self.users + self.movies))
        r = np.zeros(self.training_size)
        i = 0
        for row in self.training_indices:
            A[i, int(row[0])] = 1
            A[i, int(row[1]) + self.users] = 1
            r[i] = matrix_R[row[0], row[1]] - self.r_avg
            i += 1
        return A, r

    def get_bias(self, A, y):
        # rcond to avoid numerical errors
        return np.linalg.lstsq(A, y, rcond=1e-3)

    def get_training_size(self):
        return int(self.ratings_number * TRAINING_FRACTION)

    def get_baseline_matrix(self):
        r_baseline = np.zeros((self.users, self.movies))
        for user in range(self.users):
            for movie in range(self.movies):
                r_sum = self.r_avg + self.bias[user] + self.bias[movie + self.users]
                # crop values
                if self.mode == Mode.BASELINE:
                    if r_sum < 1.0:
                        r_sum = 1.0
                    if r_sum > 5.0:
                        r_sum = 5.0
                r_baseline[user, movie] = r_sum
        # round to the nearest integer
        return r_baseline

    def get_rmse_training(self, training_set):
        training_sum = 0.0
        for user in range(self.users):
            for movie in range(self.movies):
                if training_set[user, movie] != 0:
                    training_sum += (
                        np.rint(self.baseline_matrix[user, movie])
                        - training_set[user, movie]
                    ) ** 2
        training = m.sqrt(1.0 / self.ratings_number * training_sum)
        return np.around(training, decimals=3)

    def get_rmse_test(self, test_set, source):
        test_sum = 0.0
        for rating in test_set:
            test_sum += (np.rint(source[rating[0] - 1, rating[1] - 1]) - rating[2]) ** 2
        test = m.sqrt(1.0 / len(test_set) * test_sum)
        return np.around(test, decimals=3)

    # to be called from main program
    def calculate_absolute_errors(self, test_set, source):
        filename = "abs_errors_" + self.mode.value + ".png"
        # plot a histogram
        hist_data = [
            (abs(test_set[i][2] - source[test_set[i][0] - 1, test_set[i][1] - 1]))
            for i in range(len(test_set))
        ]
        hist, bins = np.histogram(hist_data, bins=range(10))
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align="center", width=0.7)
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.title(
            "Histogram of the distribution of the absolute errors for "
            + self.mode.value
            + " predictor\n"
        )
        plt.grid(True)
        plt.savefig(filename)
        return [x for x in hist if x > 0]

    def get_difference_matrix(self, training_data):
        # fill with special values
        diff_matrix = np.full((self.users, self.movies), NONE)
        # calculate the difference for each cell
        for user in range(self.users):
            for movie in range(self.movies):
                # if user rated the movie, calculate the difference between actual and predicted grade
                if training_data[user, movie] != 0.0:
                    diff_matrix[user, movie] = (
                        training_data[user, movie] - self.baseline_matrix[user, movie]
                    )
        return diff_matrix

    def calculate_distance_matrix(self):
        # fill with zeroes
        distance_matrix = np.zeros((self.movies, self.movies))
        for movie in range(self.movies):
            distance_matrix[movie, movie] = 0.0
            # iterate over top triangle of the matrix
            for candidate in range(movie + 1, self.movies):
                movie_a = []
                movie_b = []
                for user in range(self.users):
                    # append movies if user rated both of them
                    if (
                        self.difference_matrix[user, movie] != NONE
                        and self.difference_matrix[user, candidate] != NONE
                    ):
                        movie_a.append(self.difference_matrix[user, movie])
                        movie_b.append(self.difference_matrix[user, candidate])
                # calculate cosine coefficient distance or 0
                distance_matrix[movie, candidate] = (
                    1.0 - distance.cosine(movie_a, movie_b)
                    if len(movie_a) * len(movie_b) > 0
                    else 0.0
                )
                # get bottom triangle by symmetry
                distance_matrix[candidate, movie] = distance_matrix[movie, candidate]
        return distance_matrix

    def get_improved_matrix(self, training_data):
        improved_matrix = np.zeros((self.users, self.movies))
        sim = 0.0
        for movie in range(self.movies):
            # choose 2 closest neighbours, based on distance matrix
            neighbours = self.find_best_neighbours(movie, NEIGHBOURS_NUMBER)
            for user in range(self.users):
                similarity = self.get_similarity(training_data, user, movie, neighbours)
                temp_val = self.baseline_matrix[user, movie] + similarity / 10.0
                sim += similarity
                # crop values
                if temp_val < 1.0:
                    temp_val = 1.0
                elif temp_val > 5.0:
                    temp_val = 5.0
                improved_matrix[user, movie] = temp_val
        return improved_matrix

    def find_best_neighbours(self, movie, k):
        # returns indices of 2 closes neighbours
        n = self.movies - 1 if k >= self.movies else k
        return np.argsort([abs(x) for x in self.distance_matrix[movie]])[::-1][:n]

    def get_similarity(self, training_data, user, movie, neighbours):
        sim_denominator = 0.0
        sim_numerator = 0.0
        similarity = 0.0
        for neighbour in neighbours:
            if training_data[user, neighbour] != 0:
                sim_numerator += (
                    self.distance_matrix[movie, neighbour]
                    * self.difference_matrix[user, neighbour]
                )
                sim_denominator += abs(self.distance_matrix[movie, neighbour])
        if sim_denominator != 0:
            similarity += sim_numerator * 1.0 / sim_denominator
        return similarity

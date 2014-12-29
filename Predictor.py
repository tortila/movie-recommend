import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

TRAINING_FRACTION = 0.05
IMPROVED = "improved"
BASELINE = "baseline"
NONE = 999.999 # to distinguish unavailable data from 0 for improved predictor

class Predictor:

    def __init__(self, mode, training_data, test_data):
        self.mode = mode
        if self.mode == BASELINE:
            self.init_baseline(training_data, test_data)
        elif self.mode == IMPROVED:
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
        self.baseline_matrix = self.get_baseline_matrix(training_data)
        if self.mode == BASELINE:
            self.rmse_training = self.get_rmse_training(training_data)
            self.rmse_test = self.get_rmse_test(test_data)

    def init_improved(self, training_data, test_data):
        self.init_baseline(training_data, test_data)
        print "\t\t->\tbaseline initialized"
        self.difference_matrix = self.get_difference_matrix(training_data)
        print "\t\t->\tdiff matrix calculated"
        self.distance_matrix = self.calculate_distance_matrix()
        print "\t\t->\tdistance matrix calculated"
        self.improved_matrix = self.get_improved_matrix(training_data)
        print "\t\t->\timproved matrix calculated"
        self.rmse_test = self.get_rmse_test(test_data)

    # construct matrix A (N rows, M columns) where N is the number of training data points and M is number of (users + movies)
    def construct_a_matrix(self, matrix_R):
        # select training set of matrix R
        np.random.shuffle(self.training_indices)
        self.training_indices = self.training_indices[:self.training_size]
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

    def get_baseline_matrix(self, umr):
        r_baseline = np.zeros((self.users, self.movies))
        for user in range(self.users):
            for movie in range(self.movies):
                r_sum = self.r_avg + self.bias[user] + self.bias[user + movie]
                # crop values - only for baseline predictor
                if self.mode == BASELINE:
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
                    training_sum += (np.rint(self.baseline_matrix[user, movie]) - training_set[user, movie]) ** 2
        training = m.sqrt(1.0 / self.ratings_number * training_sum)
        return np.around(training, decimals = 3)

    def get_rmse_test(self, test_set):
        test_sum = 0.0
        source = self.baseline_matrix
        if self.mode == IMPROVED:
            source = self.improved_matrix
        for rating in test_set:
            test_sum += (np.rint(source[rating[0] - 1, rating[1] - 1]) - rating[2]) ** 2
        test = m.sqrt(1.0 / len(test_set) * test_sum)
        return np.around(test, decimals = 3)

    # to be called from main program
    def calculate_absolute_errors(self, test_set):
        source = self.baseline_matrix
        filename = "abs_errors_" + self.mode + ".png"
        if self.mode == IMPROVED:
            source = self.improved_matrix
        # plot a histogram
        hist_data = [(abs(test_set[i][2] - source[test_set[i][0] - 1, test_set[i][1] - 1])) for i in range(len(test_set))]
        hist, bins = np.histogram(hist_data, bins = range(10))
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align = "center", width = 0.7)
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.title("Histogram of the distribution of the absolute errors for " + self.mode + " predictor\n")
        plt.grid(True)
        plt.savefig(filename)
        return [x for x in hist if x > 0]

    def get_difference_matrix(self, training_data):
        diff_matrix = np.full((self.users, self.movies), NONE)
        # calculate the difference for each cell
        for user in range(self.users):
            for movie in range(self.movies):
                if training_data[user, movie] != 0:
                    diff_matrix[user, movie] = training_data[user, movie] - self.baseline_matrix[user, movie]
        # make training points unavailable
        for training_point in self.training_indices:
            diff_matrix[training_point[0], training_point[1]] = NONE
        return diff_matrix

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.movies, self.movies), dtype="float")
        # iterate over top triangle of the matrix
        for movie in range(self.movies):
            print "\t\t\tmovie:\t", movie
            for candidate in range(movie, self.movies):
                if movie == candidate:
                    distance_matrix[movie, movie] = 0
                else:
                    movie_a = []
                    movie_b = []
                    for user in range(self.users):
                        if self.difference_matrix[user, movie] != NONE and self.difference_matrix[user, candidate] != NONE:
                            movie_a.append(self.difference_matrix[user, movie])
                            movie_b.append(self.difference_matrix[user, candidate])
                    distance_matrix[movie, candidate] = distance.cosine(movie_a, movie_b)
                    # get bottom triangle by symmetry
                    distance_matrix[candidate, movie] = distance_matrix[movie, candidate]
        return distance_matrix
    
    def get_improved_matrix(self, training_data):
        improved_matrix = np.zeros((self.users, self.movies))
        for movie in range(self.movies):
            n1, n2 = self.find_best_neighbours(movie)
            for user in range(self.users):
                similarity = self.get_similarity(training_data, user, movie, n1, n2)
                improved_matrix[user, movie] = self.baseline_matrix[user, movie] + similarity
        return improved_matrix

    def find_best_neighbours(self, movie):
        neighbours = np.argsort([abs(x) for x in self.distance_matrix[movie]])[::-1]
        return neighbours[0], neighbours[1]

    def get_similarity(self, training_data, user, movie, n1, n2):
        sim_denominator = 0.0
        sim_numerator = 0.0
        similarity = 0.0
        if training_data[user, n1] != 0:
            sim_numerator += self.distance_matrix[movie, n1] * self.difference_matrix[user, n1]
            sim_denominator += abs(self.distance_matrix[movie, n1])
        if training_data[user, n2] != 0:
            sim_numerator += self.distance_matrix[movie, n2] * self.difference_matrix[user, n2]
            sim_denominator += abs(self.distance_matrix[movie, n2])
        if sim_denominator != 0:
            similarity += sim_numerator / sim_denominator
        return similarity
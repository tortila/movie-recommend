import numpy as np
import math as m

TRAINING_FRACTION = 0.05

class Predictor:

    def __init__(self, training_data, test_data):
        users = training_data.shape[0]
        movies = training_data.shape[1]
        self.r_avg = 0
        self.ratings_number = np.count_nonzero(training_data)
        print "ratings:", self.ratings_number
        self.training_size = self.get_training_size()
        print "training size:", self.training_size
        self.training_indices = np.transpose(np.nonzero(training_data))
        self.matrix_A, self.vector_y = self.construct_A(training_data, users, movies)
        print "\tavg:", self.r_avg
        self.bias = self.get_bias(self.matrix_A, self.vector_y)[0]
        print "\tBIAS:\tzeros:", sum(1 for i in self.bias if i == 0), "\tmax:", max(self.bias), "\tmin:", min(self.bias)
        self.basiline_matrix = self.get_baseline_matrix(training_data)
        self.rmse_training, self.rmse_test = self.get_rmse(training_data, test_data)
        print "\trmse:\n\t\ton training set:", self.rmse_training, "\n\t\ton test set:", self.rmse_test

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
        return r_baseline

    def get_rmse(self, training_set, test_set):
        training_sum = 0.0
        test_sum = 0.0
        test_entries = len(test_set)
        print "test points:", test_entries
        # compute rmse for training set
        for coord in self.training_indices:
            training_sum += (self.basiline_matrix[coord[0], coord[1]] - training_set[coord[0], coord[1]]) ** 2
        training = m.sqrt(1.0 / self.ratings_number * training_sum)
        # compute rmse for test set
        for rating in test_set:
            test_sum += (self.basiline_matrix[rating[0] - 1, rating[1] - 1] - rating[2]) ** 2
        print "test_sum:", test_sum
        test = m.sqrt(1.0 / test_entries * test_sum)
        return training, test

import numpy as np
import math as m

TRAINING_FRACTION = 0.05

class Predictor:

    def __init__(self, parsed_data, users, movies):
        self.r_avg = 0
        self.ratings_number = np.count_nonzero(parsed_data)
        print "ratings:", self.ratings_number
        self.training_size = self.get_training_size()
        print "training size:", self.training_size
        self.training_indices = np.transpose(np.nonzero(parsed_data))
        self.matrix_A, self.vector_y = self.construct_A(parsed_data, users, movies)
        print "A.shape:", self.matrix_A.shape, "y.shape:", self.vector_y.shape
        self.bias = self.get_bias(self.matrix_A, self.vector_y)[0]
        print "bias:", self.bias.T, "with size:", len(self.bias.T)
        print "zeros:", sum(1 for i in self.bias if i == 0), "max:", max(self.bias), "min:", min(self.bias)
        self.rmse = self.get_rmse(users, movies)
        print "rmse:", self.rmse

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
        print "avg:", self.r_avg

        # construct the matrix A
        A = np.zeros((self.training_size, users + movies))
        r = np.zeros(self.training_size)
        i = 0
        for row in self.training_indices:
            A[i, int(row[0])] = 1
            A[i, int(row[1]) + users] = 1
            r[i] = matrix_R[row[0], row[1]] - self.r_avg
            i += 1
        print "5 elems of y:", r[:5]
        return A, r.T

    def get_bias(self, A, y):
        return np.linalg.lstsq(A, y)
    
    def get_training_size(self):
        return int(self.ratings_number * TRAINING_FRACTION)

    def get_rmse(self, users, movies):
        return m.sqrt(1.0 / self.training_size * sum(m.pow(self.r_avg + self.bias[i] + self.bias[i + users], 2) for i in range(movies)))

import numpy as np

TRAINING_FRACTION = 0.05

class Predictor:

    def __init__(self, parsed_data, users, movies):
        self.r_avg = 0
        self.ratings_number = np.count_nonzero(parsed_data)
        print "ratings:", self.ratings_number
        self.training_points = int(self.ratings_number * TRAINING_FRACTION)
        print "training:", self.training_points
        self.matrix_A = self.construct_A(parsed_data, users, movies)

    # construct matrix A (N rows, M columns) where N is the number of training data points and M is number of (users + movies)
    def construct_A(self, matrix_R, users, movies):

        # select training set of matrix R
        training_indices = np.transpose(np.nonzero(matrix_R))
        np.random.shuffle(training_indices)
        training_indices = training_indices[:self.training_points]

        # calculate average rating of training set
        sum = 0
        for row in training_indices:
            sum += matrix_R[row[0], row[1]]
        self.r_avg = sum * 1.0 / self.training_points
        print "avg:", self.r_avg

        # construct the matrix A
        A = np.zeros((self.training_points, users + movies))
        i = 0
        for row in training_indices:
            A[i, int(row[0])] = 1
            A[i, int(row[1]) + users] = 1
            i += 1
        return A

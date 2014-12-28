import csv
import numpy as np

DIR = "data/"
DATA = "data"
MOVIE_NAMES = "moviename"
TEST = "test"

class Parser:

    def __init__(self, mode):
        self.users_number = 0
        self.movies_number = 0
        self.movies_file = DIR + mode + "." + MOVIE_NAMES
        self.data_file = DIR + mode + "." + DATA
        self.test_file = DIR + mode + "." + TEST
        self.titles = self.moviename_parse(self.movies_file)
        self.training_matrix = self.get_umr_matrix(self.data_parse(self.data_file))
        self.test_set = self.data_parse(self.test_file)
        print "Parser ready!"
        print self.users_number, "users and", self.movies_number, "movies."

    # for *.data and *.test files (u, m, r)
    def data_parse(self, filename):
        data = []
        with open(filename, "rb") as csv_file:
            f_reader = csv.reader(csv_file, delimiter=",")
            for row in f_reader:
                data.append([int(row[0]), int(row[1]), int(row[2])])
        self.users_number = max(map(lambda x: x[0], data))
        csv_file.close()
        return data

    def get_umr_matrix(self, raw_data):
        matrix = np.zeros((self.users_number, self.movies_number))
        for row in raw_data:
            matrix[row[0] - 1, row[1] - 1] = row[2]
        return matrix

    # for *.moviename files (m, title)
    def moviename_parse(self, filename):
        data = []
        with open(filename, "rb") as csv_file:
            f_reader = csv.reader(csv_file, delimiter=";")
            for row in f_reader:
                data.append(row[1])
                self.movies_number += 1
        csv_file.close()
        return data

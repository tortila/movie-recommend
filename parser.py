import csv
from enum import Enum
from typing import NamedTuple

import numpy as np

DIR = "data/"
DATA = "training"
MOVIE_NAMES = "moviename"
TEST = "test"


class Mode(Enum):
    BASELINE = "baseline"
    IMPROVED = "improved"


class Parser(NamedTuple):
    file_mode: str

    @classmethod
    def from_mode(cls, mode: Mode) -> "Parser":
        return Parser(file_mode=mode.value)

    @property
    def training_matrix(self):
        return self._build_umr_matrix(_data_parse(f"{DIR}/{self.file_mode}.{DATA}"))

    @property
    def test_set(self):
        return _data_parse(f"{DIR}/{self.file_mode}.{TEST}")

    def _build_umr_matrix(self, raw_data):
        users_number = max(map(lambda x: x[0], raw_data))
        titles = _moviename_parse(f"{DIR}/{self.file_mode}.{MOVIE_NAMES}")
        movies_number = len(titles)
        matrix = np.zeros((users_number, movies_number))
        for row in raw_data:
            matrix[row[0] - 1, row[1] - 1] = row[2]
        return matrix


def _moviename_parse(filename):
    """
    Method for parsing *.moviename files (m, title)
    :param filename: name of the file to parse
    :return: flat list of movie titles
    """
    data = []
    with open(filename, "r") as csv_file:
        f_reader = csv.reader(csv_file, delimiter=";")
        for row in f_reader:
            data.append(row[1])
    csv_file.close()
    return data


def _data_parse(filename):
    """
    Method for parsing *.training and *.test files (u, m, r)
    :param filename: name of the file to parse
    :return: list of lists [u, m, r]
    """
    data = []
    with open(filename, "r") as csv_file:
        f_reader = csv.reader(csv_file, delimiter=",")
        for row in f_reader:
            data.append([int(row[0]), int(row[1]), int(row[2])])
    csv_file.close()
    return data

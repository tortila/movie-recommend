import csv
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import NamedTuple, List

import numpy as np

DIR = "data"
DATA = "training"
MOVIE_NAMES = "moviename"
TEST = "test"

UmrList = List[List[int]]
UmrMatrix = np.ndarray


class Mode(Enum):
    BASELINE = "baseline"
    IMPROVED = "improved"


class Parser(NamedTuple):
    file_mode: str

    @classmethod
    def from_mode(cls, mode: Mode) -> "Parser":
        return Parser(file_mode=mode.value)

    @property
    def training_matrix(self) -> UmrMatrix:
        training_file = Path(f"{DIR}/{self.file_mode}.{DATA}")
        return self._build_umr_matrix(_data_parse(training_file))

    @property
    def test_set(self) -> UmrList:
        test_set_file = Path(f"{DIR}/{self.file_mode}.{TEST}")
        return _data_parse(test_set_file)

    def _build_umr_matrix(self, raw_data) -> UmrMatrix:
        """
        Build UMR Matrix from a list of entries.
        :param raw_data: a list of lists [u, m, r]
        :return: a 2-dimensional array A with A[u][m] = r
        """
        users_number = max(map(lambda x: x[0], raw_data))
        movienames_file = Path(f"{DIR}/{self.file_mode}.{MOVIE_NAMES}")
        titles = _moviename_parse(movienames_file)
        movies_number = len(titles)
        matrix = np.zeros((users_number, movies_number))
        for row in raw_data:
            matrix[row[0] - 1, row[1] - 1] = row[2]
        return matrix


def _moviename_parse(filename: PathLike) -> List[str]:
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


def _data_parse(filename: PathLike) -> UmrList:
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

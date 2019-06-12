from __future__ import annotations

import csv
from dataclasses import dataclass
from os import PathLike
from typing import NewType, Sequence

import numpy as np

User = NewType("User", int)
Movie = NewType("Movie", int)
Rating = NewType("Rating", int)

UMR = (User, Movie, Rating)
UmrMatrix = np.ndarray


@dataclass
class Dataset:
    records: Sequence[UMR] = ()

    @classmethod
    def from_file(cls, filename: PathLike) -> Dataset:
        """
        Method for parsing CSV files containing ratings (u, m, r)
        :param filename: path of the CSV file to parse
        :return: Dataset
        """
        data = []
        with open(filename, "r") as csv_file:
            f_reader = csv.reader(csv_file, delimiter=",")
            for row in f_reader:
                data.append((int(row[0]), int(row[1]), int(row[2])))
        return Dataset(records=data)

    def as_umr_matrix(self) -> UmrMatrix:
        """
        Build UMR Matrix from a list of entries.
        :return: a 2-dimensional array A with A[u][m] = r
        """
        users_number = self._get_max_value(0)
        movies_number = self._get_max_value(1)
        matrix = np.zeros((users_number, movies_number))
        for row in self.records:
            # normalise indices (start with 0)
            matrix[row[0] - 1, row[1] - 1] = row[2]
        return matrix

    def _get_max_value(self, position: int) -> int:
        return max(map(lambda x: x[position], self.records))

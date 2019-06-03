from pytest import fixture
from hypothesis import given, strategies as st
from hypothesis_csv.strategies import csv as csv_st

from parser import Mode, DATA, MOVIE_NAMES, TEST

MAX_VALUE = 20
integers_strategy = st.integers(min_value=1, max_value=5)
max_20_strategy = st.integers(min_value=1, max_value=MAX_VALUE)


@given(
    csv_data_small_batch=csv_st(
        columns=[integers_strategy, integers_strategy, integers_strategy], lines=20
    ),
    csv_data_large_batch=csv_st(
        columns=[max_20_strategy, max_20_strategy, integers_strategy], lines=100
    ),
)
def fake_data_files(csv_data_small_batch, csv_data_large_batch, files_path, mode):
    test_data = files_path / f"{mode}.{TEST}"
    test_data.write_text(csv_data_small_batch)
    training_data = files_path / f"{mode}.{DATA}"
    training_data.write_text(csv_data_large_batch)
    movie_data = files_path / f"{mode}.{MOVIE_NAMES}"
    movie_data.write_text(movie_titles(MAX_VALUE))


@fixture
def mock_csv_files(mocker, tmp_path):
    fake_data_files(files_path=tmp_path, mode=Mode.BASELINE.value)
    fake_data_files(files_path=tmp_path, mode=Mode.IMPROVED.value)
    mocker.patch("parser.DIR", tmp_path)


def movie_titles(n):
    return "\n".join([f"{i};Really Interesting Title {i}" for i in range(1, n + 1)])


@fixture
@given(test_data=st.lists(st.permutations(range(1, 6)), min_size=10))
def mock_parser_reading_files(mocker, test_data):
    mocker.patch("parser._data_parse", return_value=test_data)
    mocker.patch("parser._moviename_parse", return_value=movie_titles(10))

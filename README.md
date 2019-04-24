movie-recommend
===============

## About the project
TSKS11 course project

This project is published here only for the sake of sharing the knowledge. 
Unfortunately, the data that was used in the program couldn't be distributed beyond 
the participants of TSKS11 course. To show a vague idea of how the testfiles looked like, 
please see below:

* `*.training` and `*.test` - (user, movie, rating): three numbers per row, 
comma-separated, connecting the indices user-movie with the rating that the user gave for 
the movie
* `*.moviename` - (index; title; additional info): for example: 
`1; Toy Story (1995); Animation|Children's|Comedy`

All files originaly placed under `data/` directory.

## Running
#### Install dependencies
```bash
pipenv install
```
#### Run
```bash
pipenv run python movie-recommend.py {baseline|improved}
```
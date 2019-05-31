movie-recommend
===============

## About the project
TSKS11 course project

This project is published here only for the sake of sharing the knowledge. 
Unfortunately, the data that was used in the program could not be distributed beyond 
the participants of the TSKS11 course. To show a vague idea of how the test files looked like, 
please see below:

```bash
$ ls -hog data
total 4,7M
-rw-r--r-- 1  66K kwi 24 16:02 baseline.moviename
-rw-r--r-- 1 228K kwi 24 16:02 baseline.test
-rw-r--r-- 1 2,1M kwi 24 16:02 baseline.training
-rw-r--r-- 1  66K kwi 24 16:02 improved.moviename
-rw-r--r-- 1 236K kwi 24 16:02 improved.test
-rw-r--r-- 1 2,1M kwi 24 16:02 improved.training
```
All files originaly placed under `data/` directory (this program also requires that).

* `*.training` and `*.test` - (user, movie, rating): three numbers per row, 
comma-separated integers, connecting the indices user-movie with the rating that the user gave for 
the movie. As an example, row `4,1,4` indicates that user 4 gave the movie 1 a rating of 4.
* `*.moviename` - (index; title; additional info): for example: 
`1; Toy Story (1995); Animation|Children's|Comedy`

## Running
#### Install dependencies
```bash
pipenv install
```
#### Run
```bash
pipenv run python recommend.py {baseline|improved}
```
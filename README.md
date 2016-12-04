#Repeated Suvery Kalman Filter (RSK)
This project implements the repeated surveys Kalman filter of Jo Thori Lind found in the papers [here](http://folk.uio.no/jlind/papers/DP333.pdf) and [here](http://folk.uio.no/jlind/papers/surveys.pdf).  
It is based on the author's original source written for the Ox language.

## Tests
We have a few tests that check the results of our python implementation against the original Ox implementation.
To run these tests, from the root of the project execute:

```
python -m unittest
```

## Usage guide

The RSK filter is implemented in the RSK class. Initialize the class with the transition and translation matrices:
```
from rsk import RSK
rsk = RSK(transition_matrix, translation_matrix)
```
To apply the Kalman filter, call `fit` on an RSK instance:
```
rsk.fit(y, sigma, a0, Q0, Q)
```

After `fit` has been applied, the `rsk.alpha` vector becomes available. 

The 3D array `y` is the data collected from a repeated survey.  Its shape is:
```
n_periods x n_individuals x n_vars
```

## Notation guide

| Variable      | Code          | Description  |
| :------------- |:-------------| :-----|
|T|n_periods|Number of point in time measurements|
|N|n_individuals|Number of individuals|
|m|n_vars|Number of observed variables per individual per time slice|
|n|n_alpha|Length of the α vector|
|F|transition_matrix|Markov transition matrix|
|Z|translation_matrix|Translates α into group means μ|
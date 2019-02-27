This is the code developed to complete Georgia Tech CS 7641 (Machine Learning) assignment 2 (Randomized Optimization).  

The objective was to test four random search approaches:

* Random Hill Climbing
* Simulated Annealing
* Genetic Algorithm
* MIMIC

Using four problems:

* Training the weights of a neural network
* Flipflop
* Knapsack
* Travelling Salesperson

The code is broken up into a Jupyter Notebook for each problem, with basic results described.  Supporting helper functions are contained in utilities.py and utilities_hw2.py.  Search algorithms are implemented in a modified version of the mlrose package, which is included in mlrose_local and must be intalled locally (pip -e ./mlrose_local)
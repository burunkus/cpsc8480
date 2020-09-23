# Page Rank Algorithm
This code computes the ranks of a directed graph using the power-iteration method to iteratively update the rank vector until it converges or until a custom number of iteration steps. 

#### Packages needed to be installed before running the code:
•	Numpy
•	Networkx
•	Scipy
•	Pygraphviz

This program was tested using Python3.7.

### Running the code
The code starts off by reading the inputted file (.mtx ), generates an adjacency matrix, constructs a graph using the adjacency matrix and networkx. It then runs the pagerank algorithm to compute the ranks of the nodes/vertices and prints the ranks.

There are two directed graphs files of format matrix market format included in this repo:
•	pesa.mtx 
•	cage10.mtx

The name of the python code is page_rank.py. To run the code:
- To run the code till the rank converges, do:
  - python page_rank.py <filename.mxt>
  - E.g. ```python page_rank.py pesa.mtx```
  - Or ```python page_rank.py cage10.mtx```

- To run the code for a specific number of iteration, do:
  - python page_rank.py <filename.mtx> <number-of-iteration> <--num>
  - E.g. ```python page_rank.py pesa.mtx 20 --num```
  - Or ```python page_rank.py case10.mtx 50 --num```

Download [sparse graph](https://sparse.tamu.edu/)

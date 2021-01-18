# Applying Centrality Measures for Ranking Papers in Cocitation Networks
Applies centrality metrics - Betweenness, Closness, Degree and PageRank to research paper ranking

#### Packages needed to be installed before running the code:
•	Numpy
•	Networkx
•	Scipy
•	matplotlib
. feedparser

Folder contains the two datasets downloaded from [SNAP](https://snap.stanford.edu/):
•	cit-HepPh-dates.txt
•	cit-HepPh.txt
To learn more about this citation network dataset visit the dataset [link](https://snap.stanford.edu/data/cit-HepPh.html)

from these datasets various citation networks are created according to the data in cit-HepPh-dates.txt:
•	92-94.txt
•	92-96.txt
•	92-98.txt
•	92-20.txt
•	92-02.txt

From the citation datasets various co-citation network datasets were created:
•	cocitation-92-94.txt
•	cocitation-92-96.txt
•	cocitation-92-98.txt
•	cocitation-92-00.txt
•	cocitation-92-02.txt

Centrality measures are applied to the datasets to get the top ranked papers. Since the papers are ids, [paper details](https://github.com/burunkus/cpsc8480/blob/master/project/paper_details.py) and [paper name](https://github.com/burunkus/cpsc8480/blob/master/project/paper_name.py) is used to the details and title of
the papers. 

Note: running this code takes incredible amount of time because of the computation of different centrality measures especially those based on shortest path (betweenness and closeness). 

This program was tested using Python3.7.

### Running the code
The code can be run with the example matrix format file by simply typing python filename.py filename.mtx in the command line
- E.g. ```python co-citation.py```

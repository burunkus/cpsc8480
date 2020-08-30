# Graph operations using Networkx

## Packages needed to be installed before running the code:
•	Numpy
•	Networkx
•	Scipy
•	Pygraphviz

This program was tested using Python3.7.

### Running the code
Code starts off by reading the inputted file (.mtx ), generates an adjacency matrix, constructs a graph using the adjacency matrix and networkx, draws the graph using pygraphviz and saves the graph as an image to a file called graph.png. It then runs a breath first search algorithm and compares it to the networkx bfs implementation. Finally, it computes the 5 smallest eigenvalues of the graph Laplacian. The code also reports progress. 

Depending on what you want to do there are different ways to run the code.
The name of the python code is graphs.py. let us assume that the name of the matrix market file is test.mtx, then on the command line:

- The base case. i.e. Run without adding any nodes or edge attributes. It performs the tasks described above. 
  - e.g. ```python graphs.py test.mtx```
  
- To add a node attribute. Specify the node, the node attribute name or key, the node attribute value and a --node flag. 
  - e.g to add a node 10, attribute ‘time’, attribute value ‘3.2’ do:
    ```python graphs.py test.mtx 10 time 3.2 --node```
  
- To add node attribute and view nodes after addition add the --viewnodes flag:
  - e.g. ```python graphs.py test.mtx 10 time 3.2 time --node --viewnodes```
      Note: the last ‘time’ specifies that you want to view nodes with the ‘time’ attribute. 
  
- To add an edge attribute. Specify the edge (u and v), the edge attribute name or key, the edge attribute value and a --edge flag.
  - e.g. to add an edge (10 and 11), attribute ‘weight’, attribute value ‘2’ do:
  ```python graphs.py test.mtx 10 11 weight 2 --edge```
  
- To add edge attribute and view the edge attribute value just added add the --viewedge flag:
  - e.g. ```python graphs.py test.mtx 10 11 weight 2 weight --edge --viewedge```
     Note: the last ‘weight’ specifies that you want to view the edge attribute value of the edge attribute name ‘weight’, of the edge (10, 11). Which should output   2. This can be useful if you want to confirm that the attribute was added.
  
- To only add a node attribute and an edge attribute , add the --node and --edge flags.
  - e.g ```python graphs.py test.mtx 10 time 3.2 10 11 weight 2 --node --edge```
  
- To add a node attribute, edge attribute, view nodes with a certain attribute name, view the value of the just added edges’ attribute. Add the --node flag to add a node, --edge flag to add an edge, --viewnodes flag to view nodes with specific attribute name, and --viewedge flag to view the just added edge attribute value. 
 - e.g. ```python graphs.py test.mtx 10 time 3.2 time 10 11 weight 2 weight --node --edge --viewnodes --viewedges```

   This will add 10 as a node, set it’s attribute name/key to ‘time’ and set the value of the attribute name/key to 3.2 and displays nodes which have attribute  name/key ‘time’. Similary, it will add the edge 10 and 11, set it’s attribute name/key to ‘weight’, set the attribute value of the attribute name/key to 2 and display the value of the just added edge attribute which is 2. 

### Visualization
To show that this code works, I plotted a graph with 4k nodes using this code on palmetto. I also noticed that pygraphviz had trouble plotting huge graphs like 10k. Below is an image of a 4k node. This was generated with the command:

### Example 
```
python graphs.py test.mtx
```

Download [sparse graph](https://sparse.tamu.edu/)
Any graph that is too big will require the use of a GPU.
 

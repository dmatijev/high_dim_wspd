# WELL-SEPARATED PAIR  DECOMPOSITIONS FOR HIGH-DIMENSIONAL DATASETS
Well-Separated Pair Decomposition (WSPD) is a well-known geometric decomposition used forencoding distances,  introduced in 1995 by Paul B. Callahan and S. Rao Kosaraju in a seminal paper.  WSPD compresses O(n^2) pairwise distances of $n$ given points in R^dd in O(n) space for a fixed dimension d. However, the main problem with this remarkable decomposition is the ’hidden’ dependence on the dimension d, which in practice does not allow us to compute WSPD for any dimension d >2 or d >3 at best. 

Here we show how to compute WSPD for points in R^d and for any dimension d. Instead ofcomputing WSPD directly in R^d, we propose to learn a nonlinear mapping and transform the data to a lower dimensional space R^d′, d′= 2 or d′= 3, since only in such a low dimensional spaces WSPD can be efficiently computed. Furthermore, we estimate the quality of computed WSPD in the original R^d space. Our experiments show that for different synthetic and real-world datasets our approach allow that a WSPD of size O(n) is still computed for points in R^d for dimensions d much larger than two or three in practice.

We refer to our approach as NN-WSPD algorithm. 


<p align="center">
  <img src="https://github.com/dmatijev/high_dim_wspd/blob/main/wspd_dumbells.png?raw=true" width="350" >
  <img src="https://github.com/dmatijev/high_dim_wspd/blob/main/wspd_vs_nnwspd.png?raw=true" width="350" >
</p>
In both figures, the $x$-coordinate stands for the dimension, while the $y$ coordinate stands for the number of dumbbells divided by the size of all pairwise distances, i.e., $n(n-1)/2$.
 Size of the dataset is $n = 5000$. In the left figure the dumbbells 
  are computed using the WSPD algorithm by Callahan and Kosaraju, and the 
  size of WSPD (i.e., the number of dumbbells) is reported.  In the right figure
the number of dumbbells stays constant with our NN-WSPD approach even when the dimension grows. 

# ParallelMatrixMultiplication
The project is concerned with the realization of a computational core for the product between two matrices, which is then able to calculate
C ← C + AB
where A is an m × k matrix and B is a k × n matrix. For the input matrices, two main cases were considered:
1. Square matrices m = n = k;
2. Rectangular matrices m, n ≫ k; in this case k must take values
typical of the tessellation of the data k = 32, 64, 128, 156

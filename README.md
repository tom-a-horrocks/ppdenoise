# Python version of _ppdenoise_
This repository contains a Python port of Kovesi's 2D phase preserving denoising algorithm, based on the Julia implementation (https://github.com/peterkovesi/ImagePhaseCongruency.jl/blob/master/src/phasecongruency.jl). See `2D_Implementation_Test.ipynb` for an example of use and comparison with output from the original code. Note that a small bug has been fixed regarding frequency spacing in the filters, so the outputs should not be identical.

It also contains a 3D extension of the algorithm. A fixed number of orientations are used (10), representing the face normals of the top half of an icosahedron. Take a look at `3D_Implementation_Example.ipynb` for an example.

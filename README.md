# Active learning technique for quantile estimation

This code estimates a quantile of an uni-dimensional field variable of a stochastic
process. It uses the Karhunen-Loève loeve (KL) decomposition to reduce the 
dimensonality of the stochastic process and predict new outputs based on a 
surrogate model  of the uncertain variables of the KL decomposition using
Gaussian process (GP). The active learning technique looks to reduce the 
vairance of the GPs by identifying in an optimization loop the new training samples
that contribute the most for this purpose.

Two example cases are presented in the script "ActiveLearningProblem".
A simple case using the Onera Damage Model for Composite Material with Ceramix 
Matrix (ODM-CMC) and more complex launch vehicle multidisciplinary optimization
case based on the Dymos and OpenMDAO environments.

To define a new  "ActiveLearningProblem" see the ""ActiveLearningProblem.py""
script. The active learning technique is run from the "main.py" file.


## Main requirements
* dymos ( v.0.15.0 included in this repository as this code is not compatible with the version available on the original repo. To see more on this, check issue #406 in the Dymos Github page.) https://github.com/OpenMDAO/dymos
* openmdao==3.1.0
* openturns==1.17
* numpy==1.20.3
* matplotlib==3.4.3
* cma==3.1.0
* scipy==1.6.3

## Execution
Git LFS was used to compress a large file containing a dataset for Dymos. Hence zip downloads from GitHub don't work. The repo has to be cloned.
* create a new folder with path : my_path
* open CMD
* cd my_path
* git init
* git clone "link to this repo"
* run the main.py file 

## Publication
Some of the results from this project are featured in the following publication:
https://www.mdpi.com/2076-3417/12/19/10027

## Acknowledgements
This work was developed during my internship at ONERA - The French aerospace lab, and was possible thanks to guidance and help of Dr. Loïc Brevault and Dr. Mathieu Balesdent.

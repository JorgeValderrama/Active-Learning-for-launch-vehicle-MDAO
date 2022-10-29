# Active learning technique for quantile estimation

The coupling of uncertainty quantification methodologies with multidisciplinary optimization tools for the early design phase of launch vehicles is computationally intensive. This is mainly due to the strategies for multidisciplinary coupling satisfaction and the required optimal control methods for the trajectory discipline. The early design phase is characterized by a high number of input uncertain variables (e.g., specific impulse, drag coefficient) that render uncertain the output fields (e.g., the optimal speed profile as function of time and the optimal pressure distribution on aerodynamic surfaces). The output fields are comprised of a high number of correlated aleatory variables that make even more daunting the uncertainty quantification task. This work presents an Active Learning (AL) methodology for field variable quantile estimation relying on a surrogate model to reduce the computational cost. Such a surrogate model utilizes Gaussian processes, model order redcution methods and evolutionary optimization.The computed quantiles are useful to characterize flight envelops of launch vehicles. An example case is demonstrated for the quantile estimation of the resulting state variables from the Multidisciplinary Design Analysis and Optimization Proposed methods (MDAO) of a Two-Stage-To-Orbit (TSTO) launch vehicle. The methodology improves the accuracy of the An active learning methodology for the estimation of predicted quantiles and outperforms an aleatory enrichment strategy.

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

# Active learning technique for quantile estimation

This code estimates a quantile of a unidemnsional field variable on an stochastic
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


## Requirements
* dymos ( v.0.15.0 included in this repository as this code is not compatible with the version available on the original repo. To see more on this, check issue #406 in the Dymos Github page.) https://github.com/OpenMDAO/dymos
* openmdao==3.1.0

## Execution
run the main.py file

### Acknowledgements
This work was developed during my internship at ONERA - The French aerospace lab, and was possible thanks to guidance and help of Dr. Loïc Brevault and Dr. Mathieu Balesdent.
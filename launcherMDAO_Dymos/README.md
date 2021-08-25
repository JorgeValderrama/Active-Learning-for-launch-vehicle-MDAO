### Multidisciplinary optimization (MDO) of Two-Stage to Orbit (TSTO) launch vehicle

This code uses a quasi-AAO MDO formulation to optimize a TSTO Launch vehicle.
It includes 3 main disciplines: Propulsion, Mass-Sizing and trajectory.
The trajectory optimization is carried using the Legendre Gauss Lobatto Transcription.
The guidance of the launcher is done using a parameterized pitch angle guidance program.

## Requirements
dymos==0.15.0
openmdao==3.1.0

The vesion of Dymos 0.15.0 in which the code was developed is different to the Dymos 0.15.0 available in GitHub.
Hence you will need to do some modifications of the code to run it. This is related to the issue #406 in the Dymos Github page.

# Deep Flow Control
Source code for "Deep Dynamical Modeling and Control of Unsteady Fluid Flows" from NIPS 2018. A description of the individual files is given below.

![](gifs/vortex.gif)

## ```training``` Directory
* ```koopman_model.py``` - script for defining architecture of and constructing Deep Koopman models for training.
* ```train_koopman.py``` - training script for Deep Koopman models.
* ```bayes_filter.py``` - script for defining architecture of and constructing Deep Variational Bayes Filter models.
* ```train_bayes_filter.py``` - training script for Deep Variational Bayes Filter models.
* ```dataloader.py``` - script for loading and processing data prior to training.
* ```utils.py``` - contains functions for evaluating trained models.
* ```find_matrices.py``` - script to load a trained neural network model and determine the B-matrix, action normalization parameters, and goal state encoding.
* ```find_dynamics.py``` - script to load a trained neural network model and output the current state encoding and the A-matrix based on the previous sequence of observed states and actions.

Example command to train a Deep Koopman model:

```python train_koopman.py --num_filters 256 128 64 32 16 --control_input True```

## ```mpc_files``` Directory
* ```config.ini``` - example config file that defines parameters for a PyFR simulation of the 2D cylinder system. Relevant parameters that may need to be modified can be found in the ```soln-plugin-controller``` section.
* ```new.patch``` - patch that can be applied to PyFR to allow for prescribing an angular velocity on the surface of the cylinder and performing model predictive control.
* ```mesh.pyfrm``` - mesh file required to run simulation of 2D cylinder system.
* ```cyl-2d-p2-1530.pyfrs``` - solution file that can be used to initialize simulations of the 2D cylinder system.
* ```base.h5``` - snapshot of steady base flow that defines the goal state in model predictive control.
* ```loc_to_idx.json``` - JSON file containing a map from spatial locations in full CFD solutions to indices in the arrays used as neural network inputs.






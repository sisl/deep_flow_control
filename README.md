# Deep Flow Control
Source code for "Deep Dynamical Modeling and Control of Unsteady Fluid Flows" from NIPS 2018. A description of the individual files is given below.

## ```training``` Directory
* ```koopman_model.py``` - script for defining architecture of and constructing Deep Koopman models for training.
* ```train_koopman.py``` - training script for Deep Koopman models.
* ```bayes_filter.py``` - script for defining architecture of and constructing Deep Variational Bayes Filter models.
* ```train_bayes_filter.py``` - training script for Deep Variational Bayes Filter models.
* ```dataloader.py``` - script for loading and processing data prior to training.
* ```utils.py``` - contains functions for evaluating trained models.

Example command to train a Deep Koopman model:

```python train_koopman.py --num_filters 256 128 64 32 16 --control_input True```

## ```mpc_scripts``` Directory
* ```config.ini``` - example config file that defines parameters for a PyFR simulation of the 2D cylinder system.
* ```controller.py``` - script that interfaces with PyFR to generate training data and select control inputs. Depending on specifications in config file, can either apply control inputs using a predefined control law or perform model predictive control to select control inputs. Also periodically takes full solutions, extracts the relevant information needed to construct neural network inputs, and writes them to a file.
* ```loc_to_idx.pkl``` - Pickle file containing a map from spatial locations in full CFD solutions to indices in the arrays used as neural network inputs.
* ```find_matrices.py``` - script to load a trained neural network model and determine the B-matrix, action normalization parameters, and goal state encoding.
* ```find_dynamics.py``` - script to load a trained neural network model and output the current state encoding and the A-matrix based on the previous sequence of observed states and actions.

Note that many of these scripts contain paths that are specific to the machine they were run on. They will need to be updated in order to get the code to run elsewhere.






# Deep Flow Control
Source code for "Deep Dynamical Modeling and Control of Unsteady Fluid Flows" from NIPS 2018. The paper can be found [here](https://arxiv.org/pdf/1805.07472.pdf).


![](gifs/vortex.gif)


## Overview
A description of the individual files is given below.
### ```training``` Directory
* ```koopman_model.py``` - script for defining architecture of and constructing Deep Koopman models for training.
* ```train_koopman.py``` - training script for Deep Koopman models.
* ```bayes_filter.py``` - script for defining architecture of and constructing Deep Variational Bayes Filter models.
* ```train_bayes_filter.py``` - training script for Deep Variational Bayes Filter models.
* ```dataloader.py``` - script for loading and processing data prior to training.
* ```utils.py``` - contains functions for evaluating trained models.
* ```find_matrices.py``` - script to load a trained neural network model and determine the B-matrix, action normalization parameters, and goal state encoding.
* ```find_dynamics.py``` - script to load a trained neural network model and output the current state encoding and the A-matrix based on the previous sequence of observed states and actions.

### ```mpc_files``` Directory
* ```config.ini``` - example config file that defines parameters for a PyFR simulation of the 2D cylinder system. Relevant parameters that may need to be modified can be found in the ```soln-plugin-controller``` section.
* ```controller.patch``` - patch that can be applied to PyFR to allow for prescribing an angular velocity on the surface of the cylinder and performing model predictive control.
* ```mesh.pyfrm``` - mesh file required to run simulation of 2D cylinder system.
* ```cyl-2d-p2-1530.pyfrs``` - solution file that can be used to initialize simulations of the 2D cylinder system.
* ```base.h5``` - snapshot of steady base flow that defines the goal state in model predictive control.
* ```loc_to_idx.json``` - JSON file containing a map from spatial locations in full CFD solutions to indices in the arrays used as neural network inputs.

## Getting Started
Below we detail the steps required to install the necessary software, generate training data, train a Deep Koopman model, and perform model predictive control.

### Software Installation
Make sure to install [TensorFlow](https://www.tensorflow.org/install/) and [PyFR](http://www.pyfr.org). Detailed PyFR installation instructions can be found [here](http://www.hpcadvisorycouncil.com/pdf/PyFR_Best_Practices.pdf). Make sure to install PyFR v1.7.6.

Once PyFR has been installed, copy ```controller.patch``` into the top-level directory and run ```git apply controller.patch``` to modify the PyFR code in order to enable simulation with control inputs. This patch will create a file named ```controller.py``` in the ```pyfr/plugins``` directory that contains the necessary code for defining control laws and performing model predictive control.

### Generating Training Data
Once PyFR has been successfully installed, simulations can be run in order to generate training data. Move the files ```config.ini```, ```mesh.pyfrm```, ```cyl-2d-p2-1530.pyfrs```, and ```loc_to_idx.json``` to the same directory. Modify ```config.ini``` and ```controller.py``` as desired and run the command:

```pyfr restart -bcuda -p mesh.pyfrm cyl-2d-p2-1530.pyfrs config.ini```

to begin a simulation using the CUDA backend. Training data will be saved to the directory ```save_dir```, as defined in ```config.ini ```.

### Training a Model
The scripts in the ```training``` directory can be used to train a Deep Koopman model. Examine ```train_koopman.py``` to get a sense for the arguments that can be used to define the model architecture. An example command to train a Deep Koopman model is:

```python train_koopman.py --num_filters 256 128 64 32 16 --control_input True ---data_dir (data_dir)```

where ```(data_dir)``` is the directory where training data has been stored. By default checkpoints will be written to a directory named ```./checkpoints```.

### Running MPC
Once you have a trained model, modify the arguments in ```config.ini``` to perform model predictive control. In particular, you will need to set ```mpc = 1```, change ```training_path``` to be the path to the directory where the training scripts are located, modify ```checkpoint``` to correspond to the desired model checkpoint, and change ```base_flow``` to contain the correct path to the file ```base.h5```. From this point, simulations can be run with the same command used to generate the training data.



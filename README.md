

# HOSS: Hierarchical Optimization of Systems Simulations.

Copyright (C) 2021 Upinder S. Bhalla, National Centre for Biological Sciences,
Tata Institute of Fundamental Research, Bangalore, India.

All code in HOSS is licensed under GPL 3.0 or later.


## About
HOSS provides a set of methods for performing hierarchical optimization of 
signaling and other models. The key idea is that many signaling and cellular 
processes are modular and signal flow in them is hierarchical. This
makes it possible to break up large optimization problems into small modules
such that each module that is optimized depends only on its own parameters, and
the properties of upstream blocks. The upstream blocks are evaluated first, and
held fixed as one goes successively deeper into the model. Our colleagues
Radulescu and co-workers have shown mathematically that this is 
both *more efficient*, and gives *better fits*, than 'flat' optimization where 
the whole set of parameters is fitted at the same time. We have demonstrated
that this is also true in practice on messy real-world optimization problems.

The HOSS code performs simple hierarchical optimization, flat optimization,
and two multi-start methods for optimization which use hierarchical optimization
internally. The multi-start methods are even better than plain HOSS, but require 
more resources.

The structure of an optimization pipeline is defined using a configuration file
in JSON. The schema for this configuration file is provided as part of the HOSS
project.

The HOSS project has been written up and is on bioRxiv:

"Hierarchical optimization of biochemical networks"

Nisha Ann Viswan, Alexandre Tribut, Manvel Gasparyan, Ovidiu Radulescu, Upinder S. Bhalla

[https://doi.org/10.1101/2024.08.06.606818](https://doi.org/10.1101/2024.08.06.606818)

## Examples

HOSS takes a configuration file argument, and has numerous other options. 
The configuration file specifies all aspects of the optimization, notably 
a start model, a set of experiments defined in *FindSim* format, a list of 
parameters to tweak, and optional bounds on the parameters. These are all 
organized into the hierarchy chosen for the optimization.

Here is a typical command-line invocation of HOSS:

`hoss Config/D3_b2AR_hoss.json --algorithm COBYLA --outputDir OPT_D3_b2AR_COBYLA`

For examples for running HOSS, including scripts, experimental datasets and
configuration files, see the repository for generating the figures for the 
paper: [hossFigs](https://github.com/BhallaLab/hossFigs)


## Dependencies

HOSS depends on **FindSim**, **HillTau** and **MOOSE.**

[FindSim](https://github.com/BhallaLab/FindSim) is the Framework for Integrating
Neuronal Data and Signaling Models. 

FindSim defines experiments, specially biochemcial experiments, in a standard JSON
format. A FindSim file specifies:

- The design, stimuli, and experimental conditions of an experiment
- The readouts of an experiment.
- A model on which this experiment can be run.

The FindSim code does the following:
1. Reads a FindSim file, a model, and various other optional arguments. 
2. Runs the model on the experiment definition
3. Compares the model output to the experimental readouts defined in the file and returns a *score* which says how well the model output matched experiments.
4. Optionally, it modifies parameters in the model, as required by the 
	optimization routine.

It is this *score* which the HOSS script uses as the objective function for 
its optimization.

Note that FindSim is agnostic to model type, and hence so is HOSS. At present
they can work with HillTau and MOOSE models.

[HillTau](https://github.com/BhallaLab/HillTau) is a format and program for 
specifying and running abstracted models of cellular signaling. These abstracted
models retain a direct mapping to molecules, hence it is easy to use them
in optimization calculations.

[MOOSE](https://github.com/BhallaLab/moose-core) is the Multiscale Object Oriented
Simulation environment. It is good for running ODE-based signaling models
defined in the standard SBML, as well as multiscale models combining electrical
and chemical signaling.



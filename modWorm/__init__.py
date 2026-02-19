"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'
__version__ = '1.0.42'

from julia import Main

Main.eval("""

    using DifferentialEquations, OrdinaryDiffEq, Sundials, LinearAlgebra, LogExpFunctions, Interpolations, StatsBase, Statistics

    """)

from modWorm import sys_paths                  # File paths needed for simulations

from modWorm import network_params             # Predefined neural parameters for animals
from modWorm import network_dynamics           # Predefined set of equations (Python + Julia) for individual neuron dynamics (ion channels, leak channel etc)
from modWorm import network_interactions       # Predefined set of equations (Python + Julia) for neuronal interactions through connectome (synaptic, gap, etc)
from modWorm import network_simulations        # Master functions (Python + Julia) for simulating nervous system

from modWorm import muscle_body_params         # Predefined muscles/body parameters for animals
from modWorm import muscle_dynamics            # Predefined set of equations (Python + Julia) for neurons -> muscle translations
from modWorm import body_dynamics              # Predefined set of equations (Python + Julia) for viscoelastic model
from modWorm import body_simulations           # Master functions (Python + Julia) for simulating body from neural activities

from modWorm import proprioception_simulation  # Master functions (Python + Julia) for simulating nervous system + body with proprioception, supports multi-threaded simulations

from modWorm import animation                  # Functions for animating body simulations
from modWorm import utils                      # Utility functions

from modWorm import predefined_classes_nv      # Predefined set of nervous system classes 
from modWorm import predefined_classes_mb      # Predefined set of muscle/body classes
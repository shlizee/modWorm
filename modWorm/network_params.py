"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import os
import numpy as np

from modWorm import sys_paths as paths

#####################################################################################################################################################
# C.elegans(Wildtype )###############################################################################################################################
#####################################################################################################################################################

os.chdir(paths.data_dir)

class CE:

    def __init__(self):

        self.n = 279                                                                   # Number of neurons        (N),    279
        self.cell_caps = np.ones(self.n) * 0.0015                                      # Cell capacitance         (nF),   0.0015nF (1.5pF)
        self.leak_conductances = np.ones(self.n) * 0.01                                # Leak conductance         (nS),   0.01nS (10pS)
        self.leak_potentials = np.ones(self.n) * -35.0                                 # Leak potential           (mV),  -35mV
        self.gap_conductances = np.ones((self.n, self.n)) * 0.1                        # Total gap conductance    (nS),   0.1nS (100pS)
        self.syn_conductances = np.ones((self.n, self.n)) * 0.1                        # Max synaptic conductance (nS),   0.1nS (100pS)
        self.ei_map = np.load('emask_mat_v1.npy') * -48.0                              # Reversal potential       (mV),  -48mV  
        self.synaptic_rise_tau = np.ones(self.n) * (1/1.5)                             # Synaptic rise time       (s),    0.667 
        self.synaptic_fall_tau = np.ones(self.n) * (5/1.5)                             # Synaptic fall time       (s),    3.333
        self.B = np.ones(self.n) * 0.125                                               # Width of sigmoid         (mv-1), 0.125mV-1
        self.timescale = 0.001                                                         # Integration timestep     (s)     0.001s (1ms)

CE = CE()

os.chdir(paths.default_dir)
"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import os
import numpy as np
import scipy.io as sio

from scipy import interpolate
from modWorm import sys_paths as paths

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
#####################################################################################################################################################

def init_matirx_ForceTransform(mp):

    diag_A = -1 * np.add(np.reciprocal(mp[:-1]), np.reciprocal(mp[1:]))
    lower_A = np.reciprocal(mp[1:-1])
    upper_A = np.reciprocal(mp[1:-1])

    A = np.diag(diag_A) + np.diag(lower_A, -1) + np.diag(upper_A, 1)
    A_plus = np.linalg.inv(A)
    A_plus_dimcount = len(A_plus[:,0])
    A_plus_ext = np.zeros((A_plus_dimcount + 1, A_plus_dimcount + 1))
    A_plus_ext[:A_plus_dimcount, :A_plus_dimcount] = A_plus

    return A_plus_ext

#####################################################################################################################################################
# C.elegans(Wildtype )###############################################################################################################################
#####################################################################################################################################################

class CE_env:

    def __init__(self):

        self.damping = 0                                                                                                       # Force damping factor
        self.rho_f = 2                                                                                                         # Fluid density
        self.mu = 0.01                                                                                                         # Fluid viscosity

CE_env = CE_env()

class CE:

    def __init__(self):

        self.E = 600000.                                                                                                       # Young's modulus/flexural rigidity
        self.C_N = 2                                                                                                           # Drag coefficient
        self.h_num = 24                                                                                                        # Number of segments
        self.h = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1])                  # Segment lengths
        self.a = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 0.325                  # Cross-section radius a
        self.b = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 0.325                  # Cross-section radius b
        self.alpha = 1                                                                                                         # Dimensionless body scaler
        self.rho = 2                                                                                                           # Volumetric material density
        self.area = np.pi * (self.a**2)                                                                                        # Cross-section area
        self.mp = self.rho * self.area * self.h                                                                                # Body mass 
        self.w = self.a * self.alpha                                                                                           # Body width
        self.I = np.pi * (self.a**4) / 4                                                                                       # Moment of inertia motion
        self.J = np.diag(self.rho * self.I * self.h)                                                                           # Moment of inertia segment
        self.EI = self.E * self.I                                                                                              # Stiffness motion
        self.v_bar = self.E * np.pi / 8 * (self.alpha**3)                                                                      # Stiffness segment
        self.v = self.alpha * self.a**2 * self.v_bar                                                                           # Body stiffness
        self.A_plus = init_matirx_ForceTransform(self.mp)                                                                      # Force transform
        self.scaled_factor = 6000                                                                                              # Force scaling factor
        self.c = np.asarray([6*10, 0.2*(10**2), 0.5*(10**2), 10, 30, 30])                                                      # Muscle calcium factors
        self.kappa_scaling = 1                                                                                                 # Curvature scaling
        self.kappa_forcing = 0                                                                                                 # Curvature forcing

CE = CE()

class CE_animation:

    def __init__(self):

        self.h_vis = np.asarray([0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1.05, 1.05, 1.1, 1.1, 1.1,                                       # Segment lengths for visualization  
                                 1.05, 1.05, 1, 1, 1, 1, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2])                                      #
        self.h_interp = interpolate.interp1d(np.arange(0, 24), self.h_vis, axis=0, fill_value = "extrapolate")                 # Segment length interpolation
        self.diameter_scaler = 1                                                                                               # Scaling factor for circle diameters
        self.worm_seg_color = 'black'                                                                                          # Segment color
        self.trail_point = 0                                                                                                   # Segment index to be used for movement trail
        self.display_trail = False                                                                                             # Turn on/off movement trail
        self.trail_color = 'lightgrey'                                                                                         # Movement trail color
        self.trail_width = 0.5                                                                                                 # Movement trail width
        self.fps = 100                                                                                                         # Frame per second
        self.interval = 10                                                                                                     # Interval between frames (ms)
        self.display_axis = False                                                                                              # Turn on/off x and y axis
        self.facecolor = 'white'                                                                                               # Background color

CE_animation = CE_animation()

os.chdir(paths.default_dir)
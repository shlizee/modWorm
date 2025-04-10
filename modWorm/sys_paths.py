"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import os
import platform

platform = platform.system()
default_dir = os.getcwd()

if platform == 'Windows':

    main_dir = default_dir + '\\modWorm'
    data_dir = main_dir + '\\data'
    muscle_maps_dir = main_dir + '\\muscle_maps'
    presets_input_dir = main_dir + '\\presets_input'
    presets_voltage_dir = main_dir + '\\presets_voltage'
    videos_dir = default_dir + '\\created_vids'

else:

    main_dir = default_dir + '/modWorm'
    data_dir = main_dir + '/data'
    muscle_maps_dir = main_dir + '/muscle_maps'
    presets_input_dir = main_dir + '/presets_input'
    presets_voltage_dir = main_dir + '/presets_voltage'
    videos_dir = default_dir + '/created_vids'
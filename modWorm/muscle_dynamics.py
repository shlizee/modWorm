"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np

from modWorm import Main

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
#####################################################################################################################################################

def init_muscle_Map(muscle_map):

    return muscle_map

def init_muscle_MapInv(muscle_map):

    return np.linalg.pinv(muscle_map)

#####################################################################################################################################################
# COMPUTE FUNCTIONS #################################################################################################################################
#####################################################################################################################################################

def fwd_muscle_Inputs(self, V, single_step):

    if single_step == True:

        muscle_input = np.dot(self.muscle_Map, V)

        return muscle_input

    else:

        muscle_inputs = np.zeros((len(V), len(self.muscle_Map[:, 0])))

        for k in range(len(V)):

            muscle_inputs[k, :] = np.dot(self.muscle_Map, V[k, :])

        return muscle_inputs

def fwd_muscle_Calcium(self, muscle_inputs, single_step):

    if single_step == True:

        dorsal_left = muscle_inputs[0::4]
        dorsal_right = muscle_inputs[1::4]
        ventral_left = muscle_inputs[2::4]
        ventral_right = muscle_inputs[3::4]

    else:

        dorsal_left = muscle_inputs[:, 0::4]
        dorsal_right = muscle_inputs[:, 1::4]
        ventral_left = muscle_inputs[:, 2::4]
        ventral_right = muscle_inputs[:, 3::4]

    dorsal_left_positive = np.multiply(dorsal_left, dorsal_left > 0)
    dorsal_right_positive = np.multiply(dorsal_right, dorsal_right > 0)
    ventral_left_positive = np.multiply(ventral_left, ventral_left > 0)
    ventral_right_positive = np.multiply(ventral_right, ventral_right > 0)

    dorsal_sum = np.add(dorsal_left_positive, dorsal_right_positive)
    ventral_sum = np.add(ventral_left_positive, ventral_right_positive)

    muscle_calcium = np.hstack([dorsal_sum, ventral_sum])

    return muscle_calcium

def fwd_muscle_Activity(self, muscle_calcium):

    muscle_activity = np.divide(np.power(15 * muscle_calcium, 2), 1 + np.power(15 * muscle_calcium, 2))

    return muscle_activity

def fwd_muscle_Force(self, muscle_activity):

    muscle_force = self.body_ForceScaling * muscle_activity

    return muscle_force

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Julia) #########################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function fwd_muscle_Inputs(p, V, single_step)

        if single_step == true

            muscle_input_vec = p.muscle_Map * V

            return muscle_input_vec

        else

            muscle_inputs_mat = zeros(size(V)[1], size(p["muscle_Map"][:, 1])[1])

            for k = 1:size(V)[1]

                muscle_inputs_mat[k, :] = p["muscle_Map"] * V[k, :]

            end

            return muscle_inputs_mat

        end

    end

    function fwd_muscle_Calcium(p, muscle_inputs, single_step)

        if single_step == true

            dorsal_left = muscle_inputs[1:4:end]
            dorsal_right = muscle_inputs[2:4:end]
            ventral_left = muscle_inputs[3:4:end]
            ventral_right = muscle_inputs[4:4:end]

        else

            dorsal_left = muscle_inputs[:, 1:4:end]
            dorsal_right = muscle_inputs[:, 2:4:end]
            ventral_left = muscle_inputs[:, 3:4:end]
            ventral_right = muscle_inputs[:, 4:4:end]
        end

        dorsal_left_positive = dorsal_left .* (dorsal_left .> 0)
        dorsal_right_positive = dorsal_right .* (dorsal_right .> 0)
        ventral_left_positive = ventral_left .* (ventral_left .> 0)
        ventral_right_positive = ventral_right .* (ventral_right .> 0)

        dorsal_sum = dorsal_left_positive + dorsal_right_positive
        ventral_sum = ventral_left_positive + ventral_right_positive

        muscle_calcium = cat(dorsal_sum, ventral_sum, dims = !single_step + 1)

        return muscle_calcium

    end

    function fwd_muscle_Activity(p, muscle_calcium)

        muscle_activity = (15 * muscle_calcium).^2 ./ (1 .+ (15 * muscle_calcium).^2)

        return muscle_activity

    end

    function fwd_muscle_Force(p, muscle_activity, single_step)

        if single_step == true

            muscle_force = p.body_ForceScaling * muscle_activity

        else

            muscle_force = p["body_ForceScaling"] * muscle_activity

        end

        return muscle_force

    end

    """)
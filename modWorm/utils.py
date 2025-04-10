"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import os
import numpy as np
import pandas as pd
import json

from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter
from statsmodels.tsa.api import ExponentialSmoothing
from itertools import combinations, chain

from modWorm import sys_paths as paths
from modWorm import network_dynamics as n_dyn
from modWorm import network_interactions as n_inter
from modWorm import body_simulations as b_sim
from modWorm import Main

#####################################################################################################################################################
# IMPORT FUNCTIONS ##################################################################################################################################
#####################################################################################################################################################

def load_Json(filename):

    with open(filename) as content:

        content = json.load(content)

    return content

# Load connectome data from Varshney et al, 2011 with adjustments by Haspel et al, 2012
# Varshney, Lav R., et al. "Structural properties of the Caenorhabditis elegans neuronal network." PLoS computational biology 7.2 (2011): e1001066.
# Haspel, Gal, and Michael J. O'Donovan. "A perimotor framework reveals functional segmentation in the motoneuronal network controlling locomotion in Caenorhabditis elegans."
# Journal of Neuroscience 31.41 (2011): 14611-14623.

def construct_connectome_Varshney(filepath):

    conn_gap_delta = np.load(paths.data_dir + "\\conn_gap_adjust_Varshney.npy")
    conn_syn_delta = np.load(paths.data_dir + "\\conn_syn_adjust_Varshney.npy")

    conn_syn = np.zeros((279, 279))
    conn_gap = np.zeros((279, 279))
    
    Varshney_conn = pd.read_excel(filepath).to_numpy()
    Varshney_conn_S = Varshney_conn[np.where(Varshney_conn[:, 2] == 'S')[0]]
    Varshney_conn_Sp = Varshney_conn[np.where(Varshney_conn[:, 2] == 'Sp')[0]]
    Varshney_conn_R = Varshney_conn[np.where(Varshney_conn[:, 2] == 'R')[0]]
    Varshney_conn_Rp = Varshney_conn[np.where(Varshney_conn[:, 2] == 'Rp')[0]]
    
    Varshney_conn_syn_S = np.concatenate([Varshney_conn_S, Varshney_conn_Sp])
    Varshney_conn_syn_R = np.concatenate([Varshney_conn_R, Varshney_conn_Rp])
    Varshney_conn_gap = Varshney_conn[np.where(Varshney_conn[:, 2] == 'EJ')[0]]

    Varshney_names_n1n2_syn_S = Varshney_conn_syn_S[:, :2].astype('str')
    Varshney_names_n1n2_syn_R = np.fliplr(Varshney_conn_syn_R[:, :2].astype('str'))
    Varshney_names_n1n2_gap = Varshney_conn_gap[:, :2].astype('str')

    for row_ind_source in range(279):

        for col_ind_source in range(279):
    
            source_name_pair = np.array([neuron_names[row_ind_source], neuron_names[col_ind_source]])
            pre_syn_match = np.where(Varshney_names_n1n2_syn_S[:, 0] == source_name_pair[0])[0]
            post_syn_match = np.where(Varshney_names_n1n2_syn_S[:, 1] == source_name_pair[1])[0]
            pair_match = np.intersect1d(pre_syn_match, post_syn_match)
    
            if len(pair_match) != 0:
    
                conn_syn[row_ind_source, col_ind_source] = Varshney_conn_syn_S[pair_match[0], 3]

    for row_ind_source in range(279):

        for col_ind_source in range(279):
    
            source_name_pair = np.array([neuron_names[row_ind_source], neuron_names[col_ind_source]])
            pre_syn_match = np.where(Varshney_names_n1n2_syn_R[:, 0] == source_name_pair[0])[0]
            post_syn_match = np.where(Varshney_names_n1n2_syn_R[:, 1] == source_name_pair[1])[0]
            pair_match = np.intersect1d(pre_syn_match, post_syn_match)
    
            if len(pair_match) != 0:
    
                conn_syn[row_ind_source, col_ind_source] = Varshney_conn_syn_R[pair_match[0], 3]
    
    for row_ind_source in range(279):
    
        for col_ind_source in range(279):
    
            source_name_pair = np.array([neuron_names[row_ind_source], neuron_names[col_ind_source]])
            pre_syn_match = np.where(Varshney_names_n1n2_gap[:, 0] == source_name_pair[0])[0]
            post_syn_match = np.where(Varshney_names_n1n2_gap[:, 1] == source_name_pair[1])[0]
            pair_match = np.intersect1d(pre_syn_match, post_syn_match)
    
            if len(pair_match) != 0:
    
                conn_gap[row_ind_source, col_ind_source] = Varshney_conn_gap[pair_match[0], 3]

    return conn_gap + conn_gap_delta, conn_syn + conn_syn_delta

# Load connectome data from Cook et al, 2019 with adjustments by Haspel et al, 2012
# Cook, Steven J., et al. "Whole-animal connectomes of both Caenorhabditis elegans sexes." Nature 571.7763 (2019): 63-71.

def construct_connectome_Cook(filepath):

    conn_gap_delta = np.load(paths.data_dir + "\\conn_gap_adjust_Cook.npy")
    conn_syn_delta = np.load(paths.data_dir + "\\conn_syn_adjust_Cook.npy")

    conn_syn = np.zeros((279, 279))
    conn_gap = np.zeros((279, 279))
    
    Cook_conn_syn = pd.read_excel(filepath, sheet_name='hermaphrodite chemical')
    Cook_names_syn_row = Cook_conn_syn.to_numpy()[2:-1, 2]
    Cook_names_syn_col = Cook_conn_syn.to_numpy()[1, 3:-1]
    Cook_conn_syn = Cook_conn_syn.to_numpy()[2:-1, 3:-1].astype(float)
    Cook_conn_syn[np.isnan(Cook_conn_syn)] = 0
    Cook_conn_syn = Cook_conn_syn.astype(int)
    
    Cook_conn_gap = pd.read_excel(filepath, sheet_name='hermaphrodite gap jn symmetric')
    Cook_names_gap_row = Cook_conn_gap.to_numpy()[2:-1, 2]
    Cook_names_gap_col = Cook_conn_gap.to_numpy()[1, 3:-1]
    Cook_conn_gap = Cook_conn_gap.to_numpy()[2:-1, 3:-1].astype(float)
    Cook_conn_gap[np.isnan(Cook_conn_gap)] = 0
    Cook_conn_gap = Cook_conn_gap.astype(int)

    for row_ind_source in range(279):

        row_ind_target = np.where(Cook_names_syn_row == neuron_names[row_ind_source])[0][0]

        for col_ind_source in range(279):
            
            col_ind_target = np.where(Cook_names_syn_col == neuron_names[col_ind_source])[0][0]
        
            conn_syn[row_ind_source, col_ind_source] = Cook_conn_syn[row_ind_target, col_ind_target]

    for row_ind_source in range(279):
    
        row_ind_target = np.where(Cook_names_gap_row == neuron_names[row_ind_source])[0][0]
    
        for col_ind_source in range(279):
            
            col_ind_target = np.where(Cook_names_gap_col == neuron_names[col_ind_source])[0][0]
        
            conn_gap[row_ind_source, col_ind_source] = Cook_conn_gap[row_ind_target, col_ind_target]

    return conn_gap + conn_gap_delta, conn_syn + conn_syn_delta

# Load neurons-muscle mapping data from WormAtlas
# WormAtlas, Altun, Z.F., Herndon, L.A., Wolkow, C.A., Crocker, C., Lints, R. and Hall, D.H. (ed.s) 2002-2024.

def construct_muscle_map_Hall(filepath):
        
    muscle_map = pd.read_excel(filepath).to_numpy()
    muscle_map_delta = np.load(paths.muscle_maps_dir + "\\muscle_map_adjust.npy")
    
    muscle_seg_names = ['MDL', 'MDR', 'MVL', 'MVR']
    muscle_seg_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                       '21', '22', '23', '24']
    
    muscle_seg_list = []
    
    for muscle_seg_num in muscle_seg_nums:
    
        for muscle_seg_name in muscle_seg_names:
    
            muscle_seg_list.append(muscle_seg_name + muscle_seg_num)
    
    muscle_seg_list.remove('MVL24')
    
    muscle_map_modWorm = np.zeros((95, 279))
    
    for muscle_seg_ind in range(len(muscle_seg_list)):
    
        target_neuron_inds = np.where(muscle_map[:, 1] == muscle_seg_list[muscle_seg_ind])[0]
        target_neurons = muscle_map[target_neuron_inds, 0]
        source_neurons_inds = neuron_names_2_inds(target_neurons)
        source_neurons_inds_motor = np.intersect1d(source_neurons_inds, motor_group)
        source_neurons_weights = muscle_map[target_neuron_inds, -1]
        source_neurons_weights_motor = np.isin(source_neurons_inds, source_neurons_inds_motor)
    
        muscle_map_modWorm[muscle_seg_ind, source_neurons_inds_motor] = source_neurons_weights[source_neurons_weights_motor]
    
    muscle_map_modWorm = np.vstack([muscle_map_modWorm, muscle_map_modWorm[94, :]])
    muscle_map_modWorm[94, :] = 0

    return muscle_map_modWorm + muscle_map_delta

#####################################################################################################################################################
# SYSTEM DATABASES ##################################################################################################################################
#####################################################################################################################################################

os.chdir(paths.main_dir)

channel_info = load_Json('channels.json')

neurons = load_Json('neurons.json')
neurons_list = neurons['neurons']

neuron_names = []
sensory_list = []
inter_list = []
motor_list = []

for neuron in neurons_list:

    neuron_names.append(neuron['name'])

    if neuron['group'] == 'sensory':
        
        sensory_list.append(neuron['index'])

    if neuron['group'] == 'inter':
        
        inter_list.append(neuron['index'])

    if neuron['group'] == 'motor':
        
        motor_list.append(neuron['index'])

sensory_group = np.asarray(sensory_list)
inter_group = np.asarray(inter_list)        
motor_group = np.asarray(motor_list)

os.chdir(paths.default_dir)

#####################################################################################################################################################
# NV SIMULATION #####################################################################################################################################
#####################################################################################################################################################

def cap_voltage(v_vec, vmax, scaler):
    
    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))
    
    return filtered

#----------------------------------------------------------------------------------------------------------------------------------------------------

def compute_current_terms(NN, V, S):

    i_leak_mat = np.zeros(V.shape)
    i_gap_mat = np.zeros(V.shape)
    i_syn_mat = np.zeros(V.shape)

    for step in range(len(V)):

        i_leak_mat[step, :] = n_dyn.leak_current(NN, V[step, :])
        i_gap_mat[step, :] = n_inter.gap_current(NN, V[step, :])
        i_syn_mat[step, :] = n_inter.syn_current(NN, V[step, :], S[step, :])

    combined = np.stack([i_leak_mat, i_gap_mat, i_syn_mat])

    return combined

def gaussian_smoothing(y, degree):

    y_smooth = np.zeros(y.shape)

    for k in range(len(y[0, :])):
        
        y_smooth[:, k] = gaussian_filter(y[:, k], degree)
    
    return y_smooth

def continuous_transition_scaler(old, new, t, rate, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

#####################################################################################################################################################
# QUALITY OF LIFE ###################################################################################################################################
#####################################################################################################################################################

def neuron_inds_2_names(inds):
    
    names = []
    
    for ind in inds:
        
        names.append(neuron_names[ind])
        
    names = np.asarray(names)
    
    return names

def neuron_names_2_inds(names):
    
    inds = []
    
    for name in names:
        
        inds.append(np.where(np.asarray(neuron_names) == name)[0])
        
    inds = np.asarray(inds).flatten()
    
    return inds

def all_possible_combinations(candidates):
    
    combination_list = []
    
    for k in range(1, len(candidates)):
        
        c = list(combinations(candidates, k))
        
        for row in np.array(c):
            
            combination_list.append(row)
    
    return combination_list

#----------------------------------------------------------------------------------------------------------------------------------------------------

def project_v_onto_u(v, u):
    
    factor = np.divide(np.dot(u, v), np.power(np.linalg.norm(u), 2))
    projected = factor * u
    
    return projected

def compute_mean_velocity(x, y, directional_ind_1, directional_ind_2, body_ind, dt, scaling_factor):
    
    # directional vectors
    
    x_pos_components = x[:, directional_ind_1] - x[:, directional_ind_2]
    y_pos_components = y[:, directional_ind_1] - y[:, directional_ind_2]
    positional_vecs = np.vstack([x_pos_components, y_pos_components])[:, :-1]
    
    # velocity vectors using central difference
    
    x_vel_components = np.gradient(x[:, body_ind], edge_order = 2) * (scaling_factor/dt)
    y_vel_components = np.gradient(y[:, body_ind], edge_order = 2) * (scaling_factor/dt)
    velocity_vecs = np.vstack([x_vel_components, y_vel_components])
    
    computed_vels = np.zeros(len(positional_vecs[0, :]))
    computed_signs = np.zeros(len(positional_vecs[0, :]))        

    for k in range(len(positional_vecs[0, :])):
        
        projected = project_v_onto_u(velocity_vecs[:, k], positional_vecs[:, k])
        sign = np.sign(np.dot(projected, positional_vecs[:, k]))
        computed_vels[k] = sign * np.linalg.norm(velocity_vecs[:, k])
        computed_signs[k] = sign
        
    mean_velocity = np.mean(computed_vels)
    
    return computed_vels, computed_signs, mean_velocity

#----------------------------------------------------------------------------------------------------------------------------------------------------

def compute_turn_rate_wrt_t(x_vec, y_vec):

    x_vec_dot, y_vec_dot = np.gradient(x_vec, edge_order = 1), np.gradient(y_vec, edge_order = 1)
    x_vec_ddot, y_vec_ddot = np.gradient(x_vec_dot, edge_order = 1), np.gradient(y_vec_dot, edge_order = 1)

    k_numerator = (x_vec_dot * y_vec_ddot) - (y_vec_dot * x_vec_ddot)
    k_denominator = (x_vec_dot**2 + y_vec_dot**2)**(3/2)

    k = k_numerator/k_denominator
    
    radius = 1/k
    arclengths = np.sqrt(x_vec_dot**2 +  y_vec_dot**2)
    angles = np.degrees(arclengths/radius)

    return angles

#####################################################################################################################################################
# QUALITY OF LIFE FUNCTIONS (JULIA) #################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function print_progress(id, k, milestone, nsteps, verbose_period)

        if k % milestone == 0

            progress = Int(round(k / milestone * verbose_period * 100))

            print("\rid_" * string(id) * ": " * string(progress) * "% ")

        elseif k == (nsteps - 1)

            print("\rid_" * string(id) * ": " * "Done ")

        end

    end

    """)

#####################################################################################################################################################
# POST-SIMULATION ANALYSIS ##########################################################################################################################
#####################################################################################################################################################

#####################################################################################################################################################
# POST SIMULATION FUNCTIONS #########################################################################################################################
#####################################################################################################################################################

def array_or_list(ens_obj, ens_k):

    if type(ens_obj) == list:

        return ens_obj[ens_k]

    else:

        return ens_obj

def sort_process_ensemble_sols(solution_dict_ens, NN_mb_ens):

    solution_dict_ens_sorted = {}

    for ens_k in range(len(solution_dict_ens)):

        solution_dict = solution_dict_ens[ens_k]

        assert solution_dict["NN_nv_id"] == solution_dict["NN_mb_id"]
        ens_id = solution_dict["NN_nv_id"]

        NN_mb = NN_mb_ens[ens_id]
        assert NN_mb.id == ens_id

        x, y = b_sim.solve_xy(NN_mb, solution_dict["x0"], solution_dict["y0"], solution_dict["phi"])
        x_post, y_post = b_sim.postprocess_xy(NN_mb, x, y)

        solution_dict["x_solution"] = x_post
        solution_dict["y_solution"] = y_post

        solution_dict_ens_sorted[str(ens_id)] = solution_dict

    solution_dict_ens_sorted_list = []

    for ens_k in range(len(solution_dict_ens)):

        solution_dict_ens_sorted_list.append(solution_dict_ens_sorted[str(ens_k)])

    return solution_dict_ens_sorted_list
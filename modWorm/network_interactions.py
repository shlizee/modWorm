"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np
from scipy import sparse, linalg

from modWorm import Main

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
#####################################################################################################################################################

def init_network_Electrical(conn_map, conductance_map, active_mask):

    electrical_conn = np.multiply(conn_map, conductance_map)

    active_col = np.tile(active_mask, (len(conn_map), 1))
    active_row = np.transpose(active_col)

    active_mask_mat = np.multiply(active_col, active_row)
    electrical_conn = np.multiply(electrical_conn, active_mask_mat).T

    return electrical_conn

def init_network_Chemical(conn_map, conductance_map, polarity_map, active_mask):

    chemical_conn = np.multiply(conn_map, conductance_map)

    active_col = np.tile(active_mask, (len(conn_map), 1))
    active_row = np.transpose(active_col)

    active_mask_mat = np.multiply(active_col, active_row)
    chemical_conn = np.multiply(chemical_conn, active_mask_mat).T
    polarity_map = polarity_map.T

    return np.stack([chemical_conn, polarity_map])

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Python) ########################################################################################################################
#####################################################################################################################################################

def fwd_i_Electrical(self, v):

    Vrep = np.tile(v, (self.network_Size, 1))
    i_electrical = np.multiply(self.network_Electrical, np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    return i_electrical

def fwd_i_Chemical(self, v, s):

    Vrep = np.tile(v, (self.network_Size, 1))
    VsubEj = np.subtract(np.transpose(Vrep), self.network_Chemical[1])
    i_chemical = np.multiply(np.multiply(self.network_Chemical[0], np.tile(s, (self.network_Size, 1))), VsubEj).sum(axis = 1)

    return i_chemical

#----------------------------------------------------------------------------------------------------------------------------------------------------

def fwd_i_ProprioceptiveDelayed(self, NN_mb, v, s, vth, muscle_force, body_loc, k):

    if k >= self.fdb_init_Int:

        normalized_v_delay = np.subtract(v[k - self.fdb_delay_Int], vth[k - self.fdb_delay_Int])
        inferred_v = np.dot(NN_mb.muscle_MapInv, np.dot(NN_mb.muscle_Map, normalized_v_delay))

        return inferred_v

    else:

        return 0

#----------------------------------------------------------------------------------------------------------------------------------------------------

def fwd_jacobian12_Linear(self, v, s):

    Vrep = np.tile(v, (self.network_Size, 1)) # n x n
    caprep = np.tile(self.neuron_C, (self.network_Size, 1)).T

    J1_M1 = -np.multiply(self.neuron_Linear[0], np.eye(self.network_Size)) # n x n
    Ggapsumdiag = -np.diag(self.network_Electrical.sum(axis = 1)) # n x n
    J1_M2 = np.add(self.network_Electrical, Ggapsumdiag) # n x n
    J1_M3 = np.diag(np.dot(-self.network_Chemical[0], s)) # n x n
    J1 = (J1_M1 + J1_M2 + J1_M3) / caprep # n x n

    J2_M4_2 = np.subtract(self.network_Chemical[1], np.transpose(Vrep)) # n x n
    J2 = np.multiply(self.network_Chemical[0], J2_M4_2) / caprep # n x n

    return J1, J2

def fwd_jacobian34_Linear(self, v, s, vth):

    sigmoid_V = np.reciprocal(1.0 + np.exp(-self.neuron_Chemical[2]*(v - vth))) # n
    J3_1 = np.multiply(self.neuron_Chemical[0], 1 - s) # n
    J3_2 = np.multiply(self.neuron_Chemical[2], sigmoid_V) # n
    J3_3 = 1 - sigmoid_V # n
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3)) # n x n
    J4 = np.diag(np.subtract(np.multiply(-self.neuron_Chemical[0], sigmoid_V), self.neuron_Chemical[1])) # n x n

    return J3, J4

#----------------------------------------------------------------------------------------------------------------------------------------------------

def init_vth_Linear(self):

    Gcmat = np.multiply(self.neuron_Linear[0], np.eye(self.network_Size)) # n x n
    EcVec = self.neuron_Linear[1][:, np.newaxis] # n x 1

    M1 = -Gcmat # n x n
    b1 = np.multiply(self.neuron_Linear[0][:, np.newaxis], EcVec) # n x 1

    Ggapdiag = np.subtract(self.network_Electrical, np.diag(np.diag(self.network_Electrical))) # n x n
    Ggapsum = Ggapdiag.sum(axis = 1) # n
    Ggapsummat = sparse.spdiags(Ggapsum, 0, self.network_Size, self.network_Size).toarray() # n x n
    M2 = -np.subtract(Ggapsummat, Ggapdiag) # n x n

    s_eq = np.round((self.neuron_Chemical[0]/(self.neuron_Chemical[0] + 2 * self.neuron_Chemical[1])), 4) # n
    sjmat = np.vstack([s_eq] * self.network_Size) # n x n
    S_eq = s_eq[:, np.newaxis] # n x 1
    Gsyn = np.multiply(sjmat, self.network_Chemical[0]) # n x n
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn))) # n x n
    Gsynsum = Gsyndiag.sum(axis = 1) # n
    M3 = -sparse.spdiags(Gsynsum, 0, self.network_Size, self.network_Size).toarray() # n x n

    b3 = np.dot(np.multiply(self.network_Chemical[0], self.network_Chemical[1]), S_eq) # n x 1

    M = M1 + M2 + M3

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, self.network_Size)

    self.LL_vth = LL
    self.UU_vth = UU
    self.bb_vth = bb

def compute_vth_Linear(self, i_ext, correction_term = 0):

    b = np.subtract(self.bb_vth, i_ext)
    vth = linalg.solve_triangular(self.UU_vth, linalg.solve_triangular(self.LL_vth, b, lower = True, check_finite=False), check_finite=False)

    return vth + correction_term

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Julia) #########################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function fwd_i_Electrical(p, v)

        return vec(sum(p.network_Electrical .* (v .- v'), dims = 2))

    end

    function fwd_i_Chemical(p, v, s)

        return vec(sum((p.network_Chemical[1, :, :] .* s') .* (hcat(v) .- p.network_Chemical[2, :, :]), dims = 2))

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function fwd_i_ProprioceptiveDelayed(p, NN_mb, v, s, vth, muscle_force, body_loc, k)

        if k >= p.fdb_init_Int

            normalized_v_delay = v[k - p.fdb_delay_Int, :] - vth[k - p.fdb_delay_Int, :]
            inferred_v = NN_mb.muscle_MapInv * (NN_mb.muscle_Map * normalized_v_delay)

            return inferred_v

        else

            return 0

        end

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function compute_vth_Linear(p, i_ext, correction_term = 0)

        return p.UU_vth \ (p.LL_vth \ (p.bb_vth .- i_ext)) .+ correction_term

    end

    """)
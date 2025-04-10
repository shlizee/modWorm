"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np

from modWorm import Main
from modWorm import utils

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
##################################################################################################################################################### 

def init_neuron_C(capacitance):

    return capacitance

def init_neuron_Linear(conductance, leak_voltage):

    linear_params = np.vstack([conductance, leak_voltage])

    return linear_params

def init_neuron_Nonlinear(self, channel_type, neuron_inds, params_mat, added_order, initconds_mat, using_julia):

    assert type(using_julia) == bool

    neuron_inds = np.array(neuron_inds) + using_julia
    initcond = initconds_mat.T.flatten()

    if added_order == 1:

        channel_var_inds = np.arange(len(neuron_inds) * utils.channel_info[channel_type]['num_vars']) + using_julia

        assert len(channel_var_inds) == len(initcond)

        self.ic_var_length = len(channel_var_inds)
        self.ic_initcond = initcond

    else:

        channel_var_inds = np.arange(self.ic_var_length, self.ic_var_length + len(neuron_inds) * utils.channel_info[channel_type]['num_vars']) + using_julia

        assert len(channel_var_inds) == len(initcond)

        self.ic_var_length = self.ic_var_length + len(channel_var_inds)
        self.ic_initcond = np.concatenate([self.ic_initcond, initcond])

    return (neuron_inds, params_mat, channel_var_inds, initcond)

def init_neuron_Chemical(synaptic_rise_time, synaptic_fall_time, sigmoid_width):

    synaptic_activity_params = np.vstack([synaptic_rise_time, synaptic_fall_time, sigmoid_width])

    return synaptic_activity_params

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Python) ########################################################################################################################
#####################################################################################################################################################

def fwd_i_Linear(self, v):

    i_linear = np.multiply(self.neuron_Linear[0], (v - self.neuron_Linear[1]))

    return i_linear

def fwd_activity_Chemical(self, v, s, vth):

    synaptic_rise = np.multiply(np.multiply(self.neuron_Chemical[0], (1 - s)), np.reciprocal(1 + np.exp(-self.neuron_Chemical[2]*(v - vth))))
    synaptic_fall = np.multiply(self.neuron_Chemical[1], s)
    activity_synaptic = np.subtract(synaptic_rise, synaptic_fall)

    return activity_synaptic

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Simple HH-model
# Naudin, Loïs, Juan Luis Jiménez Laredo, and Nathalie Corson. "A simple model of nonspiking neurons." Neural Computation 34.10 (2022): 2075-2101.

def fwd_i_SimpleCubic(self, v):

    neuron_inds = self.neuron_SimpleCubic[0]
    vsub, params = v[neuron_inds], self.neuron_SimpleCubic[1]

    dv_SC = -(params[:, 0] * vsub**3 + params[:, 1] * vsub**2 + params[:, 2] * vsub + params[:, 3])/params[:, 4]

    return dv_SC

def iv_SimpleCubic(params, v):

    I = params[:, 0] * v**3 + params[:, 1] * v**2 + params[:, 2] * v + params[:, 3]

    return I

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Spiking AWA HH-model
# Liu, Qiang, et al. "C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials." Cell 175.1 (2018): 57-70.

def fwd_i_SHK1_AWA(self, v, u_ic):

    neuron_inds = self.neuron_SHK1_AWA[0]
    vsub = v[neuron_inds]

    u_ind_reshaped = np.reshape(u_ic[self.neuron_SHK1_AWA[2]], (len(neuron_inds), 5)) # n x 5
    w, bk, slo, slo2, kb = u_ind_reshaped[:, 0], u_ind_reshaped[:, 1], u_ind_reshaped[:, 2], u_ind_reshaped[:, 3], u_ind_reshaped[:, 4] # n, n
    params = self.neuron_SHK1_AWA[1] # n x p

    xinf = 0.5*(1 + np.tanh((vsub-params[:, 2])/params[:, 3]))
    yinf = 0.5*(1 + np.tanh((vsub-params[:, 18])/params[:, 19]))
    zinf = 0.5*(1 + np.tanh((vsub-params[:, 14])/params[:, 15]))
    qinf = 0.5*(1 + np.tanh((vsub-params[:, 12])/params[:, 13]))
    pinf = 0.5*(1 + np.tanh((vsub-params[:, 21])/params[:, 22]))

    gkt = params[:, 25] + (params[:, 26] - params[:, 25]) * 0.5 * (1 + np.tanh(vsub - params[:, 23]) / params[:, 24])
    kir = -np.log(1 + np.exp(-0.2 * (vsub - params[:, 1] - params[:, 4]))) / 0.2 + params[:, 4]

    dv_SHK1_1 = (params[:, 0] * w + params[:, 10] * slo2 + params[:, 7] * slo + params[:, 9] + params[:, 6] * yinf * (1 - bk) + params[:, 8] * kb)
    dv_SHK1_2 = vsub - params[:, 1]
    dv_SHK1 = dv_SHK1_1 * dv_SHK1_2 + params[:, 5] * kir

    dw = (xinf - w) / params[:, 11]
    dbk = (yinf - bk) / params[:, 20]
    dslo = (zinf - slo) / params[:, 16]
    dslo2 = (qinf - slo2) / params[:, 17]
    dkb = (pinf - kb) / gkt
    du = np.concatenate([dw, dbk, dslo, dslo2, dkb])

    return dv_SHK1, du

def fwd_i_EGL19_AWA(self, v, u_ic):

    neuron_inds = self.neuron_EGL19_AWA[0]
    vsub = v[neuron_inds]

    u_ind_reshaped = np.reshape(u_ic[self.neuron_EGL19_AWA[2]], (len(neuron_inds), 2)) # n x 2
    c1, c2 = u_ind_reshaped[:, 0], u_ind_reshaped[:, 1] # n, n
    params = self.neuron_EGL19_AWA[1] # n x p

    minf = 0.5*(1 + np.tanh((vsub-params[:, 2])/params[:, 3]))
    winf = 0.5*(1 + np.tanh((vsub-params[:, 4])/params[:, 5]))

    tau = 1 / np.cosh((vsub - params[:, 6]) / (2 * params[:, 7]))

    dv_EGL19 = params[:, 0] * (c1 + params[:, 11] * c2) * (vsub - params[:, 1])
    dc1 = (minf * winf / params[:, 8] - c1) / params[:, 9] - minf * winf * c2 / (params[:, 8] * params[:, 9]) - c1 / (2 * params[:, 10] * tau) + c2 / (2 * params[:, 10] * tau)
    dc2 = (c1 - c2) / (2 * params[:, 10] * tau)
    du = np.concatenate([dc1, dc2])

    return dv_EGL19, du

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Julia) #########################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function fwd_i_Linear(p, v)

        return p.neuron_Linear[1, :] .* (v .- p.neuron_Linear[2, :])

    end

    function fwd_activity_Chemical(p, v, s, vth)

        return (p.neuron_Chemical[1, :] .* (1 .- s)) .* logistic.(p.neuron_Chemical[3, :] .* (v .- vth)) .- (p.neuron_Chemical[2, :] .* s)

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Simple HH-model
# Naudin, Loïs, Juan Luis Jiménez Laredo, and Nathalie Corson. "A simple model of nonspiking neurons." Neural Computation 34.10 (2022): 2075-2101.

    function fwd_i_SimpleCubic(p, v)

        neuron_inds = p.neuron_SimpleCubic[1]
        vsub, params = v[neuron_inds], p.neuron_SimpleCubic[2] # n, n, n x p

        dv_SC = params[:, 1] .* vsub.^3 .+ params[:, 2] .* vsub.^2 .+ params[:, 3] .* vsub .+ params[:, 4]

        return dv_SC

    end

    function iv_SimpleCubic(v, params)

        I = params[:, 1] .* v.^3 .+ params[:, 2] .* v.^2 .+ params[:, 3] .* v .+ params[:, 4]

        return I

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Hybrid HH-model
# Naudin, Loïs, et al. "Systematic generation of biophysically detailed models with generalization capability for non-spiking neurons." PloS one 17.5 (2022): e0268380.

    function fwd_i_CaP(p, v, u_ic)

        neuron_inds = p.neuron_CaP[1] 
        vsub, mCa, params = v[neuron_inds], u_ic[p.neuron_CaP[3]], p.neuron_CaP[2] # n, n, n, n x p

        dv_CaP = params[:, 1] .* mCa .* (vsub .- params[:, 2]) #gCa, ECa
        dmCa = (logistic.((vsub .- params[:, 3]) ./ params[:, 4]) .- mCa) ./ params[:, 5] #VmCa, kmCa, tmCa 

        return dv_CaP, dmCa

    end

    function fwd_i_Kir(p, v)

        neuron_inds = p.neuron_Kir[1]
        vsub, params = v[neuron_inds], p.neuron_Kir[2] # n, n, n x p

        dv_Kir = params[:, 1] .* logistic.((vsub .- params[:, 2]) ./ params[:, 3]) .* (vsub .- params[:, 4]) # gKir, VKir, kKir, EKir

        return dv_Kir

    end

    function fwd_i_Kt(p, v, u_ic)

        neuron_inds = p.neuron_Kt[1]
        vsub = v[neuron_inds] # n, n

        u_ind_reshaped = reshape(u_ic[p.neuron_Kt[3]], (length(neuron_inds), 2)) # n x 2
        mK, hk = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2] # n, n
        params = p.neuron_Kt[2] # n x p

        dv_Kt = params[:, 1] .* mK .* hK .* (vsub - params[:, 2]) # gK, Ek
        dmK = (logistic.((vsub .- params[:, 3]) ./ params[:, 4]) .- mK) ./ params[:, 5] # VmK, kmK, tmK
        dhK = (logistic.((vsub .- params[:, 6]) ./ params[:, 7]) .- hK) ./ params[:, 8] # VhK, khK, thK

        return dv_Kt, [dmK; dhK]

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Spiking AWA HH-model
# Liu, Qiang, et al. "C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials." Cell 175.1 (2018): 57-70.

    function fwd_i_SHK1_AWA(p, v, u_ic)

        neuron_inds = p.neuron_SHK1_AWA[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_SHK1_AWA[3]], (length(neuron_inds), 5)) # n x 5
        w, bk, slo, slo2, kb = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2], u_ind_reshaped[:, 3], u_ind_reshaped[:, 4], u_ind_reshaped[:, 5] # n, n
        params = p.neuron_SHK1_AWA[2] # n x p

        xinf = 0.5*(1 .+ tanh.((vsub.-params[:, 3])./params[:, 4]))
        yinf = 0.5*(1 .+ tanh.((vsub.-params[:, 19])./params[:, 20]))
        zinf = 0.5*(1 .+ tanh.((vsub.-params[:, 15])./params[:, 16]))
        qinf = 0.5*(1 .+ tanh.((vsub.-params[:, 13])./params[:, 14]))
        pinf = 0.5*(1 .+ tanh.((vsub.-params[:, 22])./params[:, 23]))

        gkt = params[:, 26] .+ (params[:, 27] .- params[:, 26]) .* 0.5 .* (1 .+ tanh.(vsub .- params[:, 24]) ./ params[:, 25])
        kir = .-log.(1 .+ exp.(.-0.2 .* (vsub .- params[:, 2] .- params[:, 5]))) ./ 0.2 .+ params[:, 5]

        dv_SHK1_1 = (params[:, 1] .* w .+ params[:, 11] .* slo2 .+ params[:, 8] .* slo .+ params[:, 10] .+ params[:, 7] .* yinf .* (1 .- bk) .+ params[:, 9] .* kb)
        dv_SHK1_2 = vsub .- params[:, 2]
        dv_SHK1 = dv_SHK1_1 .* dv_SHK1_2 .+ params[:, 6] .* kir

        dw = (xinf - w) ./ params[:, 12]
        dbk = (yinf - bk) ./ params[:, 21]
        dslo = (zinf - slo) ./ params[:, 17]
        dslo2 = (qinf - slo2) ./ params[:, 18]
        dkb = (pinf - kb) ./ gkt

        return dv_SHK1, [dw; dbk; dslo; dslo2; dkb]

    end

    function fwd_i_EGL19_AWA(p, v, u_ic)

        neuron_inds = p.neuron_EGL19_AWA[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_EGL19_AWA[3]], (length(neuron_inds), 2)) # n x 2
        c1, c2 = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2] # n, n
        params = p.neuron_EGL19_AWA[2] # n x p

        minf = 0.5*(1 .+ tanh.((vsub.-params[:, 3])./params[:, 4]))
        winf = 0.5*(1 .+ tanh.((vsub.-params[:, 5])./params[:, 6]))

        tau = 1 ./ cosh.((vsub .- params[:, 7]) ./ (2 .* params[:, 8]))

        dv_EGL19 = params[:, 1] .* (c1 .+ params[:, 12] .* c2) .* (vsub .- params[:, 2])
        dc1 = (minf .* winf ./ params[:, 9] .- c1) ./ params[:, 10] .- minf .* winf .* c2 ./ (params[:, 9] .* params[:, 10]) .- c1 ./ (2 .* params[:, 11] .* tau) .+ c2 ./ (2 .* params[:, 11] .* tau)
        dc2 = (c1 .- c2) ./ (2 .* params[:, 11] .* tau)

        return dv_EGL19, [dc1; dc2]

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Spiking AVL HH-model
# Jiang, Jingyuan, et al. "C. elegans enteric motor neurons fire synchronized action potentials underlying the defecation motor program." Nature communications 13.1 (2022): 2783.

    function fwd_i_UNC2_AVL(p, v, u_ic)

        neuron_inds = p.neuron_UNC2_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_UNC2_AVL[3]], (length(neuron_inds), 2)) # (n x 2, )
        m, h = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2] # (n,) (n,)
        params = p.neuron_UNC2_AVL[2] # n x p

        dv_UNC2 = params[:, 1] .* m.^2 .* h .* (vsub - params[:, 2])

        m_alpha = params[:, 3] .* (vsub .- params[:, 4]) ./ (1 .- exp.(-(vsub .- params[:, 4]) ./ params[:, 5]))
        m_beta  = params[:, 6] .* exp.(-(vsub .- params[:, 7]) ./ params[:, 8])
        h_alpha = params[:, 9] .* exp.(-(vsub .- params[:, 10]) ./ params[:, 11])
        h_beta  = params[:, 12] ./ (1 .+ exp.(-(vsub .- params[:, 13]) ./ params[:, 14]))
        dm = m_alpha .* (1 .- m) .- m_beta .* m
        dh = h_alpha .* (1 .- h) .- h_beta .* h

        return dv_UNC2, [dm; dh]

    end

    function fwd_i_EGL19_AVL(p, v, u_ic)

        neuron_inds = p.neuron_EGL19_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_EGL19_AVL[3]], (length(neuron_inds), 2)) # n x 2
        m, h = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2] # n, n
        params = p.neuron_EGL19_AVL[2] # n x p

        dv_EGL19 = params[:, 1] .* m .* h .* (vsub .- params[:, 2])

        tau_m = params[:, 3] .* exp.(-((vsub .- params[:, 4]) ./ params[:, 5]).^2) .+ params[:, 6] .* exp.(-((vsub .- params[:, 7]) ./ params[:, 8]).^2) .+ params[:, 9]
        tau_h = params[:, 10] .* (params[:, 11] ./ (1 .+ exp.((vsub .- params[:, 12]) ./ params[:, 13])) .+ params[:, 14] ./ (1 .+ exp.((vsub .- params[:, 15]) ./ params[:, 16])) .+ params[:, 17])
        m_inf = 1 ./ (1 .+ exp.(-(vsub .- params[:, 18]) ./ params[:, 19]))
        h_inf = (params[:, 20] ./ (1 .+ exp.(-(vsub .- params[:, 21]) ./ params[:, 22])) .+ params[:, 23]) .* (params[:, 24] ./ (1 .+ exp.((vsub .- params[:, 25]) ./ params[:, 26])) .+ params[:, 27])
        dm = (m_inf .- m) ./ tau_m
        dh = (h_inf .- h) ./ tau_h

        return dv_EGL19, [dm; dh]

    end

    function fwd_i_CCA1_AVL(p, v, u_ic)

        neuron_inds = p.neuron_CCA1_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_CCA1_AVL[3]], (length(neuron_inds), 2)) # n x 2
        m, h = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2] # n, n
        params = p.neuron_CCA1_AVL[2] # n x p

        dv_CCA1 = params[:, 1] .* m.^2 .* h .* (vsub .- params[:, 2])

        m_inf = 1 ./ (1 .+ exp.(-(vsub .- params[:, 3]) ./ params[:, 4]))
        h_inf = 1 ./ (1 .+ exp.( (vsub .- params[:, 5]) ./ params[:, 6]))
        tau_m = params[:, 7] ./ (1 .+ exp.(-(vsub .- params[:, 8]) ./ params[:, 9])) .+ params[:, 10]
        tau_h = params[:, 11] ./ (1 .+ exp.( (vsub .- params[:, 12]) ./ params[:, 13])) .+ params[:, 14]
        dm = (m_inf .- m) ./ tau_m
        dh = (h_inf .- h) ./ tau_h

        return dv_CCA1, [dm; dh]

    end

    function fwd_i_SHL1_AVL(p, v, u_ic)

        neuron_inds = p.neuron_SHL1_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_SHL1_AVL[3]], (length(neuron_inds), 3)) # n x 3
        m, hf, hs = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2], u_ind_reshaped[:, 3] # n, n
        params = p.neuron_SHL1_AVL[2] # n x p

        dv_SHL1 = params[:, 1] .* m.^3 .* (0.7 .* hf .+ 0.3 .* hs) .* (vsub .- params[:, 2])

        tau_m  = params[:, 3] ./ (exp.(-(vsub - params[:, 4]) ./ params[:, 5]) .+ exp.((vsub .- params[:, 6]) ./ params[:, 7])) .+ params[:, 8]
        tau_hf = params[:, 9] ./ (1 .+ exp.((vsub .- params[:, 10]) ./ params[:, 11])) .+ params[:, 12]
        tau_hs = params[:, 13] ./ (1 .+ exp.((vsub .- params[:, 14]) ./ params[:, 15])) .+ params[:, 16]
        m_inf  = 1 ./ (1 .+ exp.(-(vsub .- params[:, 17]) ./ params[:, 18]))
        h_inf = 1 ./ (1 .+ exp.( (vsub .- params[:, 19]) ./ params[:, 20]))
        dm = (m_inf - m) ./ tau_m
        dhf = (h_inf - hf) ./ tau_hf
        dhs = (h_inf - hs) ./ tau_hs

        return dv_SHL1, [dm; dhf; dhs]

    end

    function fwd_i_EGL36_AVL(p, v, u_ic)

        neuron_inds = p.neuron_EGL36_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_EGL36_AVL[3]], (length(neuron_inds), 3)) # n x 3
        mf, mm, ms = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2], u_ind_reshaped[:, 3] # n, n
        params = p.neuron_EGL36_AVL[2] # n x p

        dv_EGL36 = params[:, 1] .* (0.31 .* mf .+ 0.36 .* mm .+ 0.39 .* ms) .* (vsub .- params[:, 2])

        m_inf = 1 ./ (1 .+ exp.(-(vsub .- params[:, 3]) ./ params[:, 4]))
        tau_mf = params[:, 5]
        tau_mm = params[:, 6]
        tau_ms = params[:, 7]
        dmf = (m_inf - mf) ./ tau_mf
        dmm = (m_inf - mm) ./ tau_mm
        dms = (m_inf - ms) ./ tau_ms

        return dv_EGL36, [dmf; dmm; dms]

    end

    function fwd_i_EXP2_AVL(p, v, u_ic)

        neuron_inds = p.neuron_EXP2_AVL[1]
        vsub = v[neuron_inds]

        u_ind_reshaped = reshape(u_ic[p.neuron_EXP2_AVL[3]], (length(neuron_inds), 4)) # n x 3
        C1, C2, C3, O = u_ind_reshaped[:, 1], u_ind_reshaped[:, 2], u_ind_reshaped[:, 3], u_ind_reshaped[:, 4] # n, n
        params = p.neuron_EXP2_AVL[2] # n x p

        I_EXP2_ = 1 .- C1 .- C2 .- C3 .- O

        dv_EXP2 = params[:, 1] .* O .* (vsub .- params[:, 2])

        alpha_1 = params[:, 3] .* exp.(params[:, 4] .* vsub)
        beta_1 = params[:, 5] .* exp.(-params[:, 6] .* vsub)
        K_f = params[:, 7]
        K_b = params[:, 8]
        alpha_2 = params[:, 9] .* exp.(params[:, 10] .* vsub)
        beta_2 = params[:, 11] .* exp.(-params[:, 12] .* vsub)
        alpha_i = params[:, 13] .* exp.(params[:, 14] .* vsub)
        beta_i = params[:, 15] .* exp.(-params[:, 16] .* vsub)
        alpha_i2 = params[:, 17] .* exp.(params[:, 18] .* vsub)
        psi = beta_2 .* beta_i .* alpha_i2 ./ (alpha_2 .* alpha_i)
        dC1 = beta_1 .* C2 .- alpha_1 .* C1
        dC2 = alpha_1 .* C1 .+ K_b .* C3 .- (beta_1 .+ K_f) .* C2
        dC3 = K_f .* C2 .+ psi .* I_EXP2_ .+ beta_2 .* O .- (K_b .+ alpha_i2 .+ alpha_2) .* C3
        dO  = beta_i .* I_EXP2_ .+ alpha_2 .* C3 .- (beta_2 .+ alpha_i) .* O

        return dv_EXP2, [dC1; dC2; dC3; dO]

    end

    function fwd_i_NCA_AVL(p, v)

        neuron_inds = p.neuron_NCA_AVL[1]
        vsub, params = v[neuron_inds], p.neuron_NCA_AVL[2] # n, n, n x p

        dv_NCA = params[:, 1] .* (vsub .- params[:, 2])

        return dv_NCA

    end

    """)
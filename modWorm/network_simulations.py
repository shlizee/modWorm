"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np
from scipy import integrate, interpolate
from collections import namedtuple

from modWorm import Main
from modWorm import network_interactions as n_inter
from modWorm import utils

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
#####################################################################################################################################################

def init_Initcond(NN, custom_initcond = None):

    if custom_initcond != None:

        return custom_initcond

    else:

        if NN.network_Size == 1:

            voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, 1)

            if 'ic_var_length' in dir(NN):

                channel_initcond = NN.ic_initcond
                full_initcond = np.concatenate([voltage_initcond, channel_initcond])

            else:

                full_initcond = np.concatenate([voltage_initcond])

        else:

            voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, NN.network_Size)
            synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, NN.network_Size)

            if 'ic_var_length' in dir(NN):

                channel_initcond = NN.ic_initcond
                full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, channel_initcond])

            else:

                full_initcond = np.concatenate([voltage_initcond, synaptic_initcond])

    return full_initcond

#####################################################################################################################################################
# MASTER FUNCTIONS ##################################################################################################################################
#####################################################################################################################################################

def run_network(NN, input_mat, extv_mat = False, interp_method = 'linear'):

    integration_prep = prep_network_integration(NN, input_mat, extv_mat, interp_method)
    solution = integrate_ode(NN, integration_prep)

    return solution

#####################################################################################################################################################
# PREPARATION FUNCTIONS #############################################################################################################################
#####################################################################################################################################################

def prep_network_integration(NN, input_mat, extv_mat, interp_method):

    nsteps = len(input_mat)
    tf = (nsteps - 1)*NN.timescale

    timepoints = np.linspace(0, tf, nsteps)
    NN.interp_i_Ext = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_method, fill_value = "extrapolate")

    if type(extv_mat) != bool:

        NN.interp_v_Ext = interpolate.interp1d(timepoints, extv_mat, axis=0, kind = interp_method, fill_value = "extrapolate")

    integration_prep = {"input_mat": input_mat,
                        "nsteps": nsteps,
                        "tf": tf}

    integration_prep = namedtuple('integration_prep', integration_prep.keys())(*integration_prep.values())

    return integration_prep

def integrate_ode(NN, integration_prep):

    t_eval = np.linspace(0, integration_prep.tf, integration_prep.nsteps)

    if NN.network_Size == 1:

        sol = integrate.solve_ivp(fun = NN.forward_Network, 
                                  t_span = [0, integration_prep.tf],
                                  y0 = NN.initcond, 
                                  method = 'BDF',
                                  t_eval = t_eval,
                                  rtol = 1e-8, atol = 1e-8)

        v_sol = sol.y[0, :]
        cv_sol = sol.y[1:, :]

        return {"v_solution": v_sol,
                "ch_solution": cv_sol}

    else:

        if 'forward_network_Jacobian' in dir(NN):

            sol = integrate.solve_ivp(fun = NN.forward_Network,
                                      jac = NN.forward_network_Jacobian,  
                                      t_span = [0, integration_prep.tf],
                                      y0 = NN.initcond, 
                                      method = 'BDF',
                                      t_eval = t_eval,
                                      rtol = 1e-8, atol = 1e-8)

        else:

            sol = integrate.solve_ivp(fun = NN.forward_Network,
                                      t_span = [0, integration_prep.tf],
                                      y0 = NN.initcond, 
                                      method = 'BDF',
                                      t_eval = t_eval,
                                      rtol = 1e-8, atol = 1e-8)

        v_sol = sol.y[:NN.network_Size, :].T
        s_sol = sol.y[NN.network_Size:NN.network_Size*2, :].T

        vthmat = np.zeros((integration_prep.nsteps, NN.network_Size))

        k = 0

        while k < integration_prep.nsteps:

            vthmat[k, :] = NN.compute_Vth(integration_prep.input_mat[k, :])

            k += 1

        return {"v_solution": v_sol,
                "s_solution" : s_sol,
                "v_threshold": vthmat}

#####################################################################################################################################################
# MASTER FUNCTIONS (Julia) ##########################################################################################################################
#####################################################################################################################################################

def run_network_julia_ensemble(NN_ens, input_mat_ens, extv_mat_ens = False, batch_size = 8):

    NN_ens[0].forward_Network()

    vars_NN_ens = []

    for ens_k in range(len(NN_ens)):

        NN = NN_ens[ens_k]
        input_mat = utils.array_or_list(input_mat_ens, ens_k)

        nsteps = len(input_mat)
        tf = (nsteps - 1)*NN.timescale
        timepoints = np.linspace(0, tf, nsteps)

        vars_NN = vars(NN)
        vars_NN["nsteps"] = nsteps
        vars_NN["tf"] = tf
        vars_NN["timepoints"] = timepoints
        vars_NN["input_mat"] = input_mat
        vars_NN["id"] = ens_k

        if type(extv_mat_ens) != bool:

            extv_mat = utils.array_or_list(extv_mat_ens, ens_k)
            assert len(extv_mat) == len(input_mat)

            vars_NN["extv_mat"] = extv_mat

        vars_NN_ens.append(vars_NN)

    if NN.network_Size != 1:

        NN_ens[0].compute_Vth()

    nv_solution_dict_ens = Main.run_network_ensemble(vars_NN_ens, batch_size)

    return nv_solution_dict_ens

def run_network_julia(NN, input_mat, extv_mat = False):

    NN.forward_Network()

    nsteps = len(input_mat)
    tf = (nsteps - 1)*NN.timescale
    timepoints = np.linspace(0, tf, nsteps)

    vars_NN = vars(NN)
    vars_NN["nsteps"] = nsteps
    vars_NN["tf"] = tf
    vars_NN["timepoints"] = timepoints
    vars_NN["input_mat"] = input_mat

    if type(extv_mat) != bool:

        vars_NN["extv_mat"] = extv_mat

    if NN.network_Size != 1:

        NN.compute_Vth()

    nv_solution_dict = Main.run_network(vars_NN)

    return nv_solution_dict

Main.eval("""

    function run_network_ensemble(NN_ens, batch_size)

        GC.gc()

        NN_ens = construct_NN_nv_ensemble(NN_ens)

        integration_prep_ens = prep_network_integration_nv_ensemble(NN_ens)
        integrator_ens = configure_ode_solver_nv_ensemble(NN_ens, integration_prep_ens)

        sol = integrate_ode_nv_ensemble(NN_ens, integration_prep_ens, integrator_ens, batch_size)
        return sol

    end

    function run_network(NN)

        GC.gc()

        NN = construct_NN_nv(NN)

        integration_prep = prep_network_integration_nv(NN)
        integrator = configure_ode_solver_nv(NN, integration_prep)

        sol = integrate_ode_nv(NN, integration_prep, integrator)
        return sol

    end

    """)

#####################################################################################################################################################
# PREPARATION FUNCTIONS (Julia) #####################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function construct_NN_nv_ensemble(NN_ens)

        NN_nt_ens = []

        for ens_k = 1:size(NN_ens)[1]

            NN = construct_NN_nv(NN_ens[ens_k])

            push!(NN_nt_ens, NN)

        end

        return NN_nt_ens

    end

    function construct_NN_nv(NN)

        NN["input_mat"] = [NN["input_mat"][i,:] for i in 1:size(NN["input_mat"], 1)]
        NN["interp_i_Ext"] = LinearInterpolation(NN["timepoints"], NN["input_mat"], extrapolation_bc = Line())

        if haskey(NN, "extv_mat") == true

            NN["extv_mat"] = [NN["extv_mat"][i,:] for i in 1:size(NN["extv_mat"], 1)]
            NN["interp_v_Ext"] = LinearInterpolation(NN["timepoints"], NN["extv_mat"], extrapolation_bc = Line())

        end

        if NN["network_Size"] == 1

            NN = (; (Symbol(k) => v for (k,v) in NN)...)

        else

            NN["LL_vth"] = LowerTriangular(NN["LL_vth"])
            NN["UU_vth"] = UpperTriangular(NN["UU_vth"])

            NN = (; (Symbol(k) => v for (k,v) in NN)...)

        end

        return NN

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function prep_network_integration_nv_ensemble(NN_ens)

        integration_prep_ens = []

        for ens_k = 1:size(NN_ens)[1]

            integration_prep = prep_network_integration_nv(NN_ens[ens_k])

            push!(integration_prep_ens, integration_prep)

        end

        return integration_prep_ens

    end

    function prep_network_integration_nv(NN)

        return (nsteps = NN.nsteps, tf =  NN.tf)

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function configure_ode_solver_nv_ensemble(NN_ens, integration_prep_ens)

        integrator = ODEProblem(forward_Network!, NN_ens[1].initcond, (0, integration_prep_ens[1].tf), NN_ens[1])

        function prob_func(integrator, i, repeat)

            remake(integrator, u0 = NN_ens[i].initcond, tspan = (0, integration_prep_ens[i].tf), p = NN_ens[i])

        end

        integrator_ens = EnsembleProblem(integrator, prob_func = prob_func)

        return integrator_ens

    end

    function configure_ode_solver_nv(NN, integration_prep)

        integrator = ODEProblem(forward_Network!, NN.initcond, (0, integration_prep.tf), NN)

        return integrator

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function integrate_ode_nv_ensemble(NN_ens, integration_prep_ens, integrator_ens, batch_size)

        nv_solution_dict_ens = []

        sol_ens = solve(integrator_ens, CVODE_BDF(), EnsembleThreads(), trajectories = size(NN_ens)[1], batch_size = batch_size, 
                        saveat = NN_ens[1].timescale, reltol = 1e-8, abstol = 1e-8, save_everystep = false)

        for ens_k in 1:size(NN_ens)[1]

            if NN_ens[1].network_Size == 1

                v_sol = sol_ens[ens_k][1,:]'
                cv_sol = sol_ens[ens_k][2:end,:]'

                sol = Dict([("v_solution", v_sol)
                            ("ch_solution", cv_sol)
                            ("NN_id", NN_ens[ens_k].id)])

                push!(nv_solution_dict_ens, sol)

            else

                vthmat = zeros(integration_prep_ens[ens_k].nsteps, NN_ens[ens_k].network_Size)

                v_sol = sol_ens[ens_k][1:NN_ens[ens_k].network_Size,:]'
                s_sol = sol_ens[ens_k][NN_ens[ens_k].network_Size+1:NN_ens[ens_k].network_Size*2,:]'

                for k = 1:integration_prep_ens[ens_k].nsteps
              
                    vthmat[k, :] = compute_Vth(NN_ens[ens_k], NN_ens[ens_k].input_mat[k])
              
                end

                sol = Dict([("v_solution", v_sol)
                            ("s_solution", s_sol)
                            ("v_threshold", vthmat)
                            ("NN_id", NN_ens[ens_k].id)])

                push!(nv_solution_dict_ens, sol)

            end

        end

        return nv_solution_dict_ens

    end

    function integrate_ode_nv(NN, integration_prep, integrator)

        sol = solve(integrator, CVODE_BDF(), saveat = NN.timescale, reltol = 1e-8, abstol = 1e-8, save_everystep = false)

        if NN.network_Size == 1

            v_sol = sol[1,:]'
            cv_sol = sol[2:end,:]'

            return Dict([("v_solution", v_sol)
                         ("ch_solution", cv_sol)])

        else

            vthmat = zeros(integration_prep.nsteps, NN.network_Size)

            v_sol = sol[1:NN.network_Size,:]'
            s_sol = sol[NN.network_Size+1:NN.network_Size*2,:]'

            for k = 1:integration_prep.nsteps
          
                vthmat[k, :] = compute_Vth(NN, NN.input_mat[k])
              
            end

            return Dict([("v_solution", v_sol)
                         ("s_solution", s_sol)
                         ("v_threshold", vthmat)])

        end

    end

    """)
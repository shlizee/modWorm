"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np
from scipy import integrate, interpolate
from collections import namedtuple

from modWorm import utils
from modWorm import Main

#####################################################################################################################################################
# CONSTRUCTORS ######################################################################################################################################
#####################################################################################################################################################

def init_Initcond(NN_mb, x_offset = 0, y_offset = 0, orientation_angle = 0):

    # positive orientation angle --> counterclockwise rotation
    # negative orientation angle --> clockwise rotation

    xinit = np.linspace(0, 2 * np.pi, NN_mb.body_SegCount + 1) + x_offset
    yinit = np.sin(xinit) + y_offset

    angle_radians, rotation_mat = compute_rotation_matrix(orientation_angle)

    xy_coords = np.vstack([xinit, yinit])
    xy_coords_transformed = np.dot(rotation_mat, xy_coords)

    x_init = xy_coords_transformed[0, :]
    y_init = xy_coords_transformed[1, :]

    phi_init = np.arcsin(np.diff(yinit)) + angle_radians

    initcond = np.zeros(52)
    initcond[0] = xinit[0]
    initcond[1] = yinit[0]
    initcond[2:26] = phi_init

    return initcond

#####################################################################################################################################################
# MASTER FUNCTIONS ##################################################################################################################################
#####################################################################################################################################################

def run_body(NN_mb, NN_nv, solution_dict_nv, stiffness = False, rtol = 2.5e-6, atol = 1e-3):

    integration_prep = prep_body_integration(NN_mb, NN_nv, solution_dict_nv)
    solution = integrate_ode(NN_mb, integration_prep, stiffness, rtol, atol)

    return solution

#####################################################################################################################################################
# PREPARATION FUNCTIONS #############################################################################################################################
#####################################################################################################################################################

def prep_body_integration(NN_mb, NN_nv, solution_dict_nv, interp_method = 'linear'):

    muscle_Force = NN_mb.forward_Muscles(solution_dict_nv['v_solution'], solution_dict_nv['v_threshold'])

    nsteps = len(muscle_Force)
    NN_mb.timescale = NN_nv.timescale
    tf = (nsteps - 1)*NN_mb.timescale

    timepoints = np.linspace(0, tf, nsteps)
    NN_mb.interp_muscle_Force = interpolate.interp1d(timepoints, muscle_Force, axis=0, kind = interp_method, fill_value = "extrapolate")

    integration_prep = {"nsteps": nsteps,
                        "tf": tf}

    integration_prep = namedtuple('integration_prep', integration_prep.keys())(*integration_prep.values())

    return integration_prep

def integrate_ode(NN_mb, integration_prep, stiffness, rtol, atol):

    t_eval = np.linspace(0, integration_prep.tf, integration_prep.nsteps)

    if stiffness == False:

        solver_method, rtol, atol = 'RK45', rtol, atol

    else:

        solver_method, rtol, atol = 'Radau', rtol, atol

    sol = integrate.solve_ivp(fun = NN_mb.forward_Body,
                              t_span = [0, integration_prep.tf],
                              y0 = NN_mb.initcond, 
                              method = solver_method,
                              t_eval = t_eval,
                              rtol = rtol, atol = atol)

    x0_sol = sol.y[0, :] 
    y0_sol = sol.y[1, :] 
    phi_sol = sol.y[2:26, :].T

    x, y = solve_xy(NN_mb, x0_sol, y0_sol, phi_sol)
    x_post, y_post = postprocess_xy(NN_mb, x, y)

    return {"raw_x_solution": x,
            "raw_y_solution": y,
            "raw_phi_solution": phi_sol,
            "x_solution": x_post,
            "y_solution": y_post}

#####################################################################################################################################################
# MASTER FUNCTIONS (Julia) ##########################################################################################################################
#####################################################################################################################################################

def run_body_julia_ensemble(NN_mb_ens, NN_nv_ens, solution_dict_nv_ens, stiffness = False, maxiters = 1e5, batch_size = 8, rtol = 2.5e-6, atol = 1e-3):

    NN_mb_ens[0].forward_Muscles()
    NN_mb_ens[0].forward_Body()

    vars_NN_mb_ens = []

    for ens_k in range(len(NN_mb_ens)):

        NN_mb = NN_mb_ens[ens_k]
        nsteps = len(solution_dict_nv_ens[ens_k]['v_solution'])
        tf = (nsteps - 1)*NN_nv_ens[ens_k].timescale
        timepoints = np.linspace(0, tf, nsteps)

        vars_NN_mb = vars(NN_mb)
        vars_NN_mb["timescale"] = NN_nv_ens[ens_k].timescale
        vars_NN_mb["nsteps"] = nsteps
        vars_NN_mb["tf"] = tf
        vars_NN_mb["timepoints"] = timepoints
        vars_NN_mb["muscle_Force"] = Main.forward_Muscles(vars_NN_mb, solution_dict_nv_ens[ens_k]['v_solution'], solution_dict_nv_ens[ens_k]['v_threshold'])
        vars_NN_mb["stiffness"] = stiffness
        vars_NN_mb["maxiters"] = maxiters

        vars_NN_mb_ens.append(vars_NN_mb)

    x0_sol_list, y0_sol_list, phi_sol_list = Main.run_body_ensemble(vars_NN_mb_ens, batch_size, rtol, atol)

    mb_solution_dict_ens = []

    for ens_k in range(len(NN_mb_ens)):

        x, y = solve_xy(NN_mb_ens[ens_k], x0_sol_list[ens_k], y0_sol_list[ens_k], phi_sol_list[ens_k])
        x_post, y_post = postprocess_xy(NN_mb_ens[ens_k], x, y)

        mb_sol_dict = {"raw_x_solution": x,
                       "raw_y_solution": y,
                       "raw_phi_solution": phi_sol_list[ens_k],
                       "x_solution": x_post,
                       "y_solution": y_post}

        mb_solution_dict_ens.append(mb_sol_dict)

    return mb_solution_dict_ens

def run_body_julia(NN_mb, NN_nv, solution_dict_nv, stiffness = False, maxiters = 1e5, rtol = 2.5e-6, atol = 1e-3):

    NN_mb.forward_Muscles()
    NN_mb.forward_Body()

    nsteps = len(solution_dict_nv['v_solution'])
    tf = (nsteps - 1)*NN_nv.timescale
    timepoints = np.linspace(0, tf, nsteps)

    vars_NN_mb = vars(NN_mb)
    vars_NN_mb["timescale"] = NN_nv.timescale
    vars_NN_mb["nsteps"] = nsteps
    vars_NN_mb["tf"] = tf
    vars_NN_mb["timepoints"] = timepoints
    vars_NN_mb["muscle_Force"] = Main.forward_Muscles(vars_NN_mb, solution_dict_nv['v_solution'], solution_dict_nv['v_threshold'])
    vars_NN_mb["stiffness"] = stiffness
    vars_NN_mb["maxiters"] = maxiters

    x0_sol, y0_sol, phi_sol = Main.run_body(vars_NN_mb, rtol, atol)

    x, y = solve_xy(NN_mb, x0_sol, y0_sol, phi_sol)
    x_post, y_post = postprocess_xy(NN_mb, x, y)

    mb_sol_dict = {"raw_x_solution": x,
                   "raw_y_solution": y,
                   "raw_phi_solution": phi_sol,
                   "x_solution": x_post,
                   "y_solution": y_post}

    return mb_sol_dict

Main.eval("""

    function run_body_ensemble(NN_mb_ens, batch_size, rtol, atol)

        GC.gc()

        NN_mb_ens = construct_NN_mb_ensemble(NN_mb_ens)

        integration_prep_ens = prep_network_integration_mb_ensemble(NN_mb_ens)
        integrator_ens = configure_ode_solver_mb_ensemble(NN_mb_ens, integration_prep_ens)

        x0_sol_list, y0_sol_list, phi_sol_list = integrate_ode_mb_ensemble(NN_mb_ens, integration_prep_ens, integrator_ens, batch_size, rtol, atol)
        
        return x0_sol_list, y0_sol_list, phi_sol_list

    end

    function run_body(NN_mb, rtol, atol)

        GC.gc()

        NN_mb = construct_NN_mb(NN_mb)

        integration_prep = prep_network_integration_mb(NN_mb)
        integrator = configure_ode_solver_mb(NN_mb, integration_prep)

        x0_sol, y0_sol, phi_sol = integrate_ode_mb(NN_mb, integration_prep, integrator, rtol, atol)
        
        return x0_sol, y0_sol, phi_sol

    end

    """)

#####################################################################################################################################################
# PREPARATION FUNCTIONS (Julia) #####################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function construct_NN_mb_ensemble(NN_mb_ens)

        NN_mb_nt_ens = []

        for ens_k = 1:size(NN_mb_ens)[1]

            NN_mb = construct_NN_mb(NN_mb_ens[ens_k])

            push!(NN_mb_nt_ens, NN_mb)

        end

        return NN_mb_nt_ens

    end

    function construct_NN_mb(NN_mb)

        NN_mb["muscle_Force"] = [NN_mb["muscle_Force"][i,:] for i in 1:size(NN_mb["muscle_Force"], 1)]
        NN_mb["interp_muscle_Force"] = LinearInterpolation(NN_mb["timepoints"], NN_mb["muscle_Force"], extrapolation_bc = Line())

        NN_mb = (; (Symbol(k) => v for (k,v) in NN_mb)...)

        return NN_mb

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function prep_network_integration_mb_ensemble(NN_mb_ens)

        integration_prep_ens = []

        for ens_k = 1:size(NN_mb_ens)[1]

            integration_prep = prep_network_integration_nv(NN_mb_ens[ens_k])

            push!(integration_prep_ens, integration_prep)

        end

        return integration_prep_ens

    end

    function prep_network_integration_mb(NN_mb)

        return (nsteps = NN_mb.nsteps, tf = NN_mb.tf)

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function configure_ode_solver_mb_ensemble(NN_mb_ens, integration_prep_ens)

        integrator = ODEProblem(forward_Body!, NN_mb_ens[1].initcond, (0, integration_prep_ens[1].tf), NN_mb_ens[1])

        function prob_func(integrator, i, repeat)

            remake(integrator, u0 = NN_mb_ens[i].initcond, tspan = (0, integration_prep_ens[i].tf), p = NN_mb_ens[i])

        end

        integrator_ens = EnsembleProblem(integrator, prob_func = prob_func)

        return integrator_ens

    end

    function configure_ode_solver_mb(NN_mb, integration_prep)

        integrator = ODEProblem(forward_Body!, NN_mb.initcond, (0, integration_prep.tf), NN_mb)

        return integrator

    end

#----------------------------------------------------------------------------------------------------------------------------------------------------

    function integrate_ode_mb_ensemble(NN_mb_ens, integration_prep_ens, integrator_ens, batch_size, rtol, atol)

        x0_sol_list, y0_sol_list, phi_sol_list = [], [], []

        if NN_mb_ens[1].stiffness == true

            sol_ens = solve(integrator_ens, TRBDF2(autodiff=false), EnsembleThreads(), trajectories = size(NN_mb_ens)[1], batch_size = batch_size,
                            saveat = NN_mb_ens[1].timescale, reltol = rtol, abstol = atol, save_everystep = false, maxiters = NN_mb_ens[1].maxiters)

        else

            sol_ens = solve(integrator_ens, DP5(), EnsembleThreads(), trajectories = size(NN_mb_ens)[1], batch_size = batch_size,
                            saveat = NN_mb_ens[1].timescale, reltol = rtol, abstol = atol, save_everystep = false, maxiters = NN_mb_ens[1].maxiters)

        end

        for ens_k in 1:size(NN_mb_ens)[1]

            x0_sol = sol_ens[ens_k][1, :]
            y0_sol = sol_ens[ens_k][2, :]
            phi_sol = sol_ens[ens_k][3:26, :]'

            push!(x0_sol_list, x0_sol)
            push!(y0_sol_list, y0_sol)
            push!(phi_sol_list, phi_sol)

        end

        return x0_sol_list, y0_sol_list, phi_sol_list

    end

    function integrate_ode_mb(NN_mb, integration_prep, integrator, rtol, atol)

        if NN_mb.stiffness == true

            sol = solve(integrator, TRBDF2(autodiff=false), saveat = NN_mb.timescale, reltol = rtol, abstol = atol, save_everystep = false, maxiters = NN_mb.maxiters)

        else

            sol = solve(integrator, DP5(), saveat = NN_mb.timescale, reltol = rtol, abstol = atol, save_everystep = false, maxiters = NN_mb.maxiters)

        end

        x0_sol = sol[1, :]
        y0_sol = sol[2, :]
        phi_sol = sol[3:26, :]'

        return x0_sol, y0_sol, phi_sol

    end

    """)

#####################################################################################################################################################
# COMPUTE FUNCTIONS #################################################################################################################################
#####################################################################################################################################################

def compute_rotation_matrix(theta_degree):

    theta = np.radians(theta_degree)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))

    return theta, R

#####################################################################################################################################################
# POST-PROCESSING FUNCTIONS #########################################################################################################################
#####################################################################################################################################################

def solve_xy(NN_mb, x1, y1, phi):

    radii = np.divide(NN_mb.body_SegLength, 2.)

    x_coords = np.zeros((len(phi), 24))
    y_coords = np.zeros((len(phi), 24))
    x_coords[:, 0] = x1
    y_coords[:, 0] = y1

    for k in range(1, len(NN_mb.body_SegLength)):

        k_ = k - 1

        x_coords[:, k] = (NN_mb.body_SegLength[k] / 2.) * (np.cos(phi[:, k_]) + np.cos(phi[:, k])) + x_coords[:, k_]
        y_coords[:, k] = (NN_mb.body_SegLength[k] / 2.) * (np.sin(phi[:, k_]) + np.sin(phi[:, k])) + y_coords[:, k_]

    x = np.zeros(x_coords.shape)
    y = np.zeros(y_coords.shape)

    for k in range(len(phi)):

        x[k, :] = x_coords[k, :] - 0.5 * np.multiply(NN_mb.body_SegLength, np.cos(phi[k, :]))
        y[k, :] = y_coords[k, :] - 0.5 * np.multiply(NN_mb.body_SegLength, np.sin(phi[k, :]))

    return x, y

def solve_xy_single(NN_mb, x1, y1, phi):

    radii = np.divide(NN_mb.body_SegLength, 2.)

    x_coords = np.zeros(24)
    y_coords = np.zeros(24)
    x_coords[0] = x1
    y_coords[0] = y1

    for k in range(1, len(NN_mb.body_SegLength)):

        k_ = k - 1

        x_coords[k] = (NN_mb.body_SegLength[k] / 2.) * (np.cos(phi[k_]) + np.cos(phi[k])) + x_coords[k_]
        y_coords[k] = (NN_mb.body_SegLength[k] / 2.) * (np.sin(phi[k_]) + np.sin(phi[k])) + y_coords[k_]

    x = np.subtract(x_coords, (0.5 * np.multiply(NN_mb.body_SegLength, np.cos(phi))))
    y = np.subtract(y_coords, (0.5 * np.multiply(NN_mb.body_SegLength, np.sin(phi))))

    return x, y

def postprocess_xy(NN_mb, x, y):

    interpolate_x = interpolate.interp1d(np.arange(0, NN_mb.body_SegCount), x, axis=1, fill_value = "extrapolate")
    interpolate_y = interpolate.interp1d(np.arange(0, NN_mb.body_SegCount), y, axis=1, fill_value = "extrapolate")

    expanded_x = interpolate_x(np.linspace(0, NN_mb.body_SegCount, 192))
    expanded_y = interpolate_y(np.linspace(0, NN_mb.body_SegCount, 192))

    expanded_x_smoothed = utils.gaussian_smoothing(expanded_x, 5)
    expanded_y_smoothed = utils.gaussian_smoothing(expanded_y, 5)

    return expanded_x_smoothed, expanded_y_smoothed

#####################################################################################################################################################
# POST-PROCESSING FUNCTIONS (Julia) #################################################################################################################
#####################################################################################################################################################

Main.eval("""

    function solve_xy_single(NN_mb, x1, y1, phi)

        radii = NN_mb.body_SegLength / 2

        x_coords = zeros(NN_mb.body_SegCount)
        y_coords = zeros(NN_mb.body_SegCount)
        x_coords[1] = x1
        y_coords[1] = y1

        for k = 2:size(NN_mb.body_SegLength)[1]

            k_ = k - 1

            x_coords[k] = (NN_mb.body_SegLength[k] / 2) * (cos(phi[k_]) + cos(phi[k])) + x_coords[k_]
            y_coords[k] = (NN_mb.body_SegLength[k] / 2) * (sin(phi[k_]) + sin(phi[k])) + y_coords[k_]

        end

        x = x_coords - (0.5 * (NN_mb.body_SegLength .* cos.(phi)))
        y = y_coords - (0.5 * (NN_mb.body_SegLength .* sin.(phi)))

        return x, y

    end

    """)
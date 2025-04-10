"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

import numpy as np

from modWorm import Main
from modWorm import network_params as n_params
from modWorm import network_dynamics as n_dyn
from modWorm import network_interactions as n_inter
from modWorm import network_simulations as n_sim

from modWorm import muscle_body_params as mb_params
from modWorm import muscle_dynamics as m_dyn
from modWorm import body_dynamics as b_dyn
from modWorm import body_simulations as b_sim

#####################################################################################################################################################
# Pre-defined Classes ###############################################################################################################################
#####################################################################################################################################################

class CelegansWorm_MuscleBody:
    
    def __init__(self, muscle_map):
        
        """
            (REQUIRED) 
            Initialize the muscle and body class
            
            Similar to nervous system, define parameters for propagating muscle and body dynamics
            self.initcond is required. self.timescale is set by the nervous system class.
        
        """
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic rod body model
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self)

    def forward_Muscles(self, V, Vth):
        
        """
            (REQUIRED)
            Evaluate muscle states necessary for body dynamics given the network voltage (V) and resting voltage (Vth)
            Pararmeters
            
            (with proprioceptive/sensory feedback) -> Parameters and output span single timestep
            (without proprioceptiv/sensory feedback) -> Parameters and output span multiple timesteps (THIS EXAMPLE)
            
            V : NumPy array of shape (timepoints, self.network_Size) or (self.network_Size, )
                Network voltage state spanning single or multiple timesteps
                    
            Vth : NumPy array of shape (timepoints, self.network_Size) or (self.network_Size, )
                  Network resting voltage state spanning single or multiple timesteps
                    
            Out : NumPy array of shape (timepoints, self.network_Size) or (self.network_Size, )
                  Muscle state spanning single or multiple timesteps
            
        """
        
        normalized_V = V - Vth
        
        # Use pre-built function for C.elegans translating voltage states to muscle forces
        # If using these function: (without proprioceptive/sensory feedback) set single_step = False
        #                          (with proprioceptive/sensory feedback) set single_step = True
        
        muscle_Inputs = m_dyn.fwd_muscle_Inputs(self, normalized_V, single_step = False)
        muscle_Calcium = m_dyn.fwd_muscle_Calcium(self, muscle_Inputs, single_step = False)
        muscle_Activity = m_dyn.fwd_muscle_Activity(self, muscle_Calcium)
        muscle_Force = m_dyn.fwd_muscle_Force(self, muscle_Activity)
        
        return muscle_Force
    
    def forward_Body(self, t, y):
        
        """
            (REQUIRED)
            Evaluate the right-hand-side function of the body dynamics ODE system
            Pararmeters
            
            t : float
                  timepoint in which the right-hand-side function of the ODE is evaluated
                    
            y : NumPy array
                  Vector of current body state (e.g., coordinates, angles, etc)
            
            (IMPORTANT)
            (with proprioceptive/sensory feedback) -> Set body force variables fR = self.fR and fL = self.fL
            (without proprioceptiv/sensory feedback) -> Set body force variables to be determined by interp_muscle_Force()
            
        """
        
        # Unravel the input y into (x,y,phi, dx,dy,dphi)
        xyphi, xyphi_dot = np.split(y, 2)
        
        # Body coordindates
        x1, y1, phi = xyphi[0], xyphi[1], xyphi[2:]
        xdot1, ydot1, phidot = xyphi_dot[0], xyphi_dot[1], xyphi_dot[2:]
        xvec, yvec, xdot, ydot = b_dyn.fwd_body_Coordinates(self, x1, y1, xdot1, ydot1, phi, phidot)
        
        # Local segment velocities
        v_tan, v_norm = b_dyn.fwd_body_Velocity(self, phi, xdot, ydot)
        
        # Body curvature
        fR = self.interp_muscle_Force(t)[:self.body_SegCount-1]
        fL = self.interp_muscle_Force(t)[self.body_SegCount:-1]
        k = b_dyn.fwd_body_Curvature(self, fR, fL)
        
        # Body forces
        Mdiff = b_dyn.fwd_body_ContactMoment(self, phi, phidot, k)
        F_N, F_T = b_dyn.fwd_body_NormalTanForce(self, v_norm, v_tan)
        Wx, Wy, Wx_diff, Wy_diff = b_dyn.fwd_body_Forces(self, phi, F_T, F_N)
        
        # Second derivatives for body coordinates/angles
        phi_ddot = b_dyn.fwd_body_D2Angles(self, phi, phidot, Mdiff, Wy_diff, Wx_diff)
        xddot_1, yddot_1 = b_dyn.fwd_body_D2Coordinates(self, phi, phidot, phi_ddot, Wx, Wy)
        
        # Concatenate body coordinates/angles into output
        xy_dot = np.asarray([xdot[0], ydot[0]])
        xyphi_dot = np.concatenate([xy_dot, phidot])
        xy_ddot = np.asarray([xddot_1, yddot_1])
        xyphi_ddot = np.concatenate([xy_ddot, phi_ddot])
        
        return np.concatenate([xyphi_dot, xyphi_ddot])

class CelegansWorm_MuscleBody_PPC:
    
    # Identical to tutorial 2, 3 except 
    # 1. forward_Muscle() parameters and output span single timestep
    # 2. forward_Body() sets body force variables fR = self.fR and fL = self.fL
    
    def __init__(self, muscle_map):
        
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic body model
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self)

    def forward_Muscles(self, v, vth):
        
        normalized_v = v - vth
        
        muscle_Inputs = m_dyn.fwd_muscle_Inputs(self, normalized_v, single_step = True)
        muscle_Calcium = m_dyn.fwd_muscle_Calcium(self, muscle_Inputs, single_step = True)
        muscle_Activity = m_dyn.fwd_muscle_Activity(self, muscle_Calcium)
        muscle_Force = m_dyn.fwd_muscle_Force(self, muscle_Activity)
        
        return muscle_Force
    
    def forward_Body(self, t, y):
        
        # Unravel the input y into (x,y,phi, dx,dy,dphi)
        xyphi, xyphi_dot = np.split(y, 2)
        
        # Body coordindates
        x1, y1, phi = xyphi[0], xyphi[1], xyphi[2:]
        xdot1, ydot1, phidot = xyphi_dot[0], xyphi_dot[1], xyphi_dot[2:]
        xvec, yvec, xdot, ydot = b_dyn.fwd_body_Coordinates(self, x1, y1, xdot1, ydot1, phi, phidot)
        
        # Local segment velocities
        v_tan, v_norm = b_dyn.fwd_body_Velocity(self, phi, xdot, ydot)
        
        # Body curvature
        k = b_dyn.fwd_body_Curvature(self, self.fR, self.fL)
        
        # Body forces
        Mdiff = b_dyn.fwd_body_ContactMoment(self, phi, phidot, k)
        F_N, F_T = b_dyn.fwd_body_NormalTanForce(self, v_norm, v_tan)
        Wx, Wy, Wx_diff, Wy_diff = b_dyn.fwd_body_Forces(self, phi, F_T, F_N)
        
        # Second derivatives for body coordinates/angles
        phi_ddot = b_dyn.fwd_body_D2Angles(self, phi, phidot, Mdiff, Wy_diff, Wx_diff)
        xddot_1, yddot_1 = b_dyn.fwd_body_D2Coordinates(self, phi, phidot, phi_ddot, Wx, Wy)
        
        # Concatenate body coordinates/angles into output
        xy_dot = np.asarray([xdot[0], ydot[0]])
        xyphi_dot = np.concatenate([xy_dot, phidot])
        xy_ddot = np.asarray([xddot_1, yddot_1])
        xyphi_ddot = np.concatenate([xy_ddot, phi_ddot])
        
        return np.concatenate([xyphi_dot, xyphi_ddot])

class CelegansWorm_MuscleBody_sg:
    
    # Identical to tutorial 4 
    
    def __init__(self, muscle_map):
        
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic body model
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self, x_offset = 60, orientation_angle=-30)

    def forward_Muscles(self, v, vth):
        
        normalized_v = v - vth
        
        muscle_Inputs = m_dyn.fwd_muscle_Inputs(self, normalized_v, single_step = True)
        muscle_Calcium = m_dyn.fwd_muscle_Calcium(self, muscle_Inputs, single_step = True)
        muscle_Activity = m_dyn.fwd_muscle_Activity(self, muscle_Calcium)
        muscle_Force = m_dyn.fwd_muscle_Force(self, muscle_Activity)
        
        return muscle_Force
    
    def forward_Body(self, t, y): # Defines the evolution of body positions (dx/dt, dy/dt) and angles (dphi/dt) at time t
        
        # Unravel the input y into (x,y,phi, dx,dy,dphi)
        xyphi, xyphi_dot = np.split(y, 2)
        
        # Body coordindates
        x1, y1, phi = xyphi[0], xyphi[1], xyphi[2:]
        xdot1, ydot1, phidot = xyphi_dot[0], xyphi_dot[1], xyphi_dot[2:]
        xvec, yvec, xdot, ydot = b_dyn.fwd_body_Coordinates(self, x1, y1, xdot1, ydot1, phi, phidot)
        
        # Local segment velocities
        v_tan, v_norm = b_dyn.fwd_body_Velocity(self, phi, xdot, ydot)
        
        # Body curvature
        k = b_dyn.fwd_body_Curvature(self, self.fR, self.fL)
        
        # Body forces
        Mdiff = b_dyn.fwd_body_ContactMoment(self, phi, phidot, k)
        F_N, F_T = b_dyn.fwd_body_NormalTanForce(self, v_norm, v_tan)
        Wx, Wy, Wx_diff, Wy_diff = b_dyn.fwd_body_Forces(self, phi, F_T, F_N)
        
        # Second derivatives for body coordinates/angles
        phi_ddot = b_dyn.fwd_body_D2Angles(self, phi, phidot, Mdiff, Wy_diff, Wx_diff)
        xddot_1, yddot_1 = b_dyn.fwd_body_D2Coordinates(self, phi, phidot, phi_ddot, Wx, Wy)
        
        # Concatenate body coordinates/angles into output
        xy_dot = np.asarray([xdot[0], ydot[0]])
        xyphi_dot = np.concatenate([xy_dot, phidot])
        xy_ddot = np.asarray([xddot_1, yddot_1])
        xyphi_ddot = np.concatenate([xy_ddot, phi_ddot])
        
        return np.concatenate([xyphi_dot, xyphi_ddot])

#----------------------------------------------------------------------------------------------------------------------------------------------------

class CelegansWorm_MuscleBody_Julia:
    
    def __init__(self, muscle_map): # Defines fluid environment and body structure
        
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic body model
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self)

    def forward_Muscles(self): # Translation of neural activity voltages to body forces in Julia
        
        Main.eval("""
        
            function forward_Muscles(p, V, Vth)
            
                normalized_V = V - Vth
                
                muscle_Inputs = fwd_muscle_Inputs(p, normalized_V, false)
                muscle_Calcium = fwd_muscle_Calcium(p, muscle_Inputs, false)
                muscle_Activity = fwd_muscle_Activity(p, muscle_Calcium)
                muscle_Force = fwd_muscle_Force(p, muscle_Activity, false)
                
                return muscle_Force
                
            end
        
            """)
    
    def forward_Body(self): # Evolution of body positions and body segment angles at time t in Julia

        Main.eval("""

            function forward_Body!(du, u, p, t)

                xyphi, xyphi_dot = u[1:2+p.body_SegCount], u[3+p.body_SegCount:end]
                
                x1, y1, phi = xyphi[1], xyphi[2], xyphi[3:end]
                xdot1, ydot1, phidot = xyphi_dot[1], xyphi_dot[2], xyphi_dot[3:end]
                xvec, yvec, xdot, ydot = fwd_body_Coordinates(p, x1, y1, xdot1, ydot1, phi, phidot)
                
                v_tan, v_norm = fwd_body_Velocity(p, phi, xdot, ydot)

                fR = p.interp_muscle_Force(t)[1:p.body_SegCount-1]
                fL = p.interp_muscle_Force(t)[p.body_SegCount+1:end-1]
                k = fwd_body_Curvature(p, fR, fL)

                Mdiff = fwd_body_ContactMoment(p, phi, phidot, k)
                F_N, F_T = fwd_body_NormalTanForce(p, v_norm, v_tan)
                Wx, Wy, Wx_diff, Wy_diff = fwd_body_Forces(p, phi, F_T, F_N)

                phi_ddot = fwd_body_D2Angles(p, phi, phidot, Mdiff, Wy_diff, Wx_diff)
                xddot_1, yddot_1 = fwd_body_D2Coordinates(p, phi, phidot, phi_ddot, Wx, Wy)

                xy_dot = [xdot[1], ydot[1]]
                xyphi_dot = [xy_dot; phidot]
                xy_ddot = [xddot_1, yddot_1]
                xyphi_ddot = [xy_ddot; phi_ddot]

                du[:] = [xyphi_dot; xyphi_ddot]

            end
        
            """)

class CelegansWorm_MuscleBody_PPC_Julia:
    
    def __init__(self, muscle_map): # Defines fluid environment and body structure
        
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic body model
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self)

    def forward_Muscles(self): # Translation of neural activity voltages to body forces in Julia
        
        Main.eval("""
        
            function forward_Muscles(p, v, vth)
            
                normalized_v = v - vth
                
                muscle_Inputs = fwd_muscle_Inputs(p, normalized_v, true)
                muscle_Calcium = fwd_muscle_Calcium(p, muscle_Inputs, true)
                muscle_Activity = fwd_muscle_Activity(p, muscle_Calcium)
                muscle_Force = fwd_muscle_Force(p, muscle_Activity, true)
                
                return muscle_Force
                
            end
        
            """)
    
    def forward_Body(self): # Evolution of body positions and body segment angles at time t in Julia

        Main.eval("""

            function forward_Body!(du, u, p, t)

                xyphi, xyphi_dot = u[1:2+p.body_SegCount], u[3+p.body_SegCount:end]
                
                x1, y1, phi = xyphi[1], xyphi[2], xyphi[3:end]
                xdot1, ydot1, phidot = xyphi_dot[1], xyphi_dot[2], xyphi_dot[3:end]
                xvec, yvec, xdot, ydot = fwd_body_Coordinates(p, x1, y1, xdot1, ydot1, phi, phidot)
                
                v_tan, v_norm = fwd_body_Velocity(p, phi, xdot, ydot)

                k = fwd_body_Curvature(p, p.fR, p.fL)

                Mdiff = fwd_body_ContactMoment(p, phi, phidot, k)
                F_N, F_T = fwd_body_NormalTanForce(p, v_norm, v_tan)
                Wx, Wy, Wx_diff, Wy_diff = fwd_body_Forces(p, phi, F_T, F_N)

                phi_ddot = fwd_body_D2Angles(p, phi, phidot, Mdiff, Wy_diff, Wx_diff)
                xddot_1, yddot_1 = fwd_body_D2Coordinates(p, phi, phidot, phi_ddot, Wx, Wy)

                xy_dot = [xdot[1], ydot[1]]
                xyphi_dot = [xy_dot; phidot]
                xy_ddot = [xddot_1, yddot_1]
                xyphi_ddot = [xy_ddot; phi_ddot]

                du[:] = [xyphi_dot; xyphi_ddot]

            end
        
            """)

class CelegansWorm_MuscleBody_sg_Julia:
    
    def __init__(self, muscle_map): # Defines fluid environment and body structure
        
        # Parameters for translating neural activities to muscles
        self.muscle_Map = muscle_map
        self.muscle_MapInv = np.linalg.pinv(self.muscle_Map)
        
        # Parameters for the visco-elastic body model
        self.body_E = mb_params.CE.E
        self.body_DragCoeff = mb_params.CE.C_N
        self.body_SegLength = mb_params.CE.h
        self.body_SegCount = mb_params.CE.h_num
        self.body_RadiusH = mb_params.CE.a
        self.body_RadiusV = mb_params.CE.b
        self.body_Area = mb_params.CE.area
        self.body_Mass = mb_params.CE.mp
        self.body_Width = mb_params.CE.w
        self.body_InertiaMotion = mb_params.CE.I
        self.body_InertiaSeg = mb_params.CE.J
        self.body_StiffMotion = mb_params.CE.EI
        self.body_StiffSeg = mb_params.CE.v_bar
        self.body_Stiff = mb_params.CE.v
        self.body_ForceTransform = mb_params.CE.A_plus
        self.body_ForceScaling = mb_params.CE.scaled_factor
        self.body_CalciumFactor = mb_params.CE.c
        self.body_CurvatureScaling = mb_params.CE.kappa_scaling
        self.body_CurvatureForcing = mb_params.CE.kappa_forcing
        
        # Parameters for the fluid environment
        self.fluid_Damping = mb_params.CE_env.damping
        self.fluid_Density = mb_params.CE_env.rho_f
        self.fluid_Viscosity = mb_params.CE_env.mu
        
        # Initial condition for the body positions and segment angles
        self.initcond = b_sim.init_Initcond(self, x_offset = 60, orientation_angle=-30)

    def forward_Muscles(self): # Translation of neural activity voltages to body forces in Julia
        
        Main.eval("""
        
            function forward_Muscles(p, v, vth)
            
                normalized_v = v - vth
                
                muscle_Inputs = fwd_muscle_Inputs(p, normalized_v, true)
                muscle_Calcium = fwd_muscle_Calcium(p, muscle_Inputs, true)
                muscle_Activity = fwd_muscle_Activity(p, muscle_Calcium)
                muscle_Force = fwd_muscle_Force(p, muscle_Activity, true)
                
                return muscle_Force
                
            end
        
            """)
    
    def forward_Body(self): # Evolution of body positions and body segment angles at time t in Julia

        Main.eval("""

            function forward_Body!(du, u, p, t)

                xyphi, xyphi_dot = u[1:2+p.body_SegCount], u[3+p.body_SegCount:end]
                
                x1, y1, phi = xyphi[1], xyphi[2], xyphi[3:end]
                xdot1, ydot1, phidot = xyphi_dot[1], xyphi_dot[2], xyphi_dot[3:end]
                xvec, yvec, xdot, ydot = fwd_body_Coordinates(p, x1, y1, xdot1, ydot1, phi, phidot)
                
                v_tan, v_norm = fwd_body_Velocity(p, phi, xdot, ydot)

                k = fwd_body_Curvature(p, p.fR, p.fL)

                Mdiff = fwd_body_ContactMoment(p, phi, phidot, k)
                F_N, F_T = fwd_body_NormalTanForce(p, v_norm, v_tan)
                Wx, Wy, Wx_diff, Wy_diff = fwd_body_Forces(p, phi, F_T, F_N)

                phi_ddot = fwd_body_D2Angles(p, phi, phidot, Mdiff, Wy_diff, Wx_diff)
                xddot_1, yddot_1 = fwd_body_D2Coordinates(p, phi, phidot, phi_ddot, Wx, Wy)

                xy_dot = [xdot[1], ydot[1]]
                xyphi_dot = [xy_dot; phidot]
                xy_ddot = [xddot_1, yddot_1]
                xyphi_ddot = [xy_ddot; phi_ddot]

                du[:] = [xyphi_dot; xyphi_ddot]

            end
        
            """)
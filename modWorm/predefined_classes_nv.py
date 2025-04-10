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

#####################################################################################################################################################
# Pre-defined Classes ###############################################################################################################################
#####################################################################################################################################################

class CelegansWorm_NervousSystem:

    def __init__(self, gap_conn, syn_conn):
        
        """
            (REQUIRED) 
            Initialize the nervous system class with desired biophysical modules
            
            This is where you define parameters for propagating nervous system dynamics
            self.network_Size, self.initcond, self.timescale are Required
            One can use pre-built modules starting with "init" to initialize specific modules as below
        
        """
        
        # Set network size (>= 1), (REQUIRED)
        self.network_Size = n_params.CE.n

        # Add capacitance parameters to neurons, dim = (n,)
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps)
        
        # Add Linear dynamics parameters (Leaky channel) to neurons, dim = (2 x n)
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials)
        
        # Add Chemical dynamics parameters (Synaptic transmission channel) to neurons, (3 x n) 
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B)

        # Add Electrical connectome between neurons, dim = (n x n)
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool'))
        
        # Add Chemical connectome between neurons, dim = (2 x n x n)
        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool'))
        
        # Set initial condition and integration timescale (seconds) (REQUIRED)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self, i_Ext):
        
        """
            (REQUIRED)
            Evaluate network-wise resting potential given the external current input
            Pararmeters
            
            i_Ext : NumPy array of shape (self.network_Size, )
                    Vector of External input current
                    
            Out : NumPy array of shape (self.network_Size, )
                  Vector of network-wise resting potential values
        """
        
        Vth = n_inter.compute_vth_Linear(self, i_Ext) # Using pre-built function
        
        return Vth

    def forward_Network(self, t, y):
        
        """
            (REQUIRED)
            Evaluate the right-hand-side function of the nervous system ODE system
            Pararmeters
            
            t : float
                  timepoint in which the right-hand-side function of the ODE is evaluated
                    
            y : NumPy array
                  Vector of current network state (e.g., voltage, synaptic variable, ion channel variables)
        """

        v, s = y[:self.network_Size], y[self.network_Size:]
        
        i_Linear = n_dyn.fwd_i_Linear(self, v)
        i_Electrical = n_inter.fwd_i_Electrical(self, v)
        i_Chemical = n_inter.fwd_i_Chemical(self, v, s)
        i_Ext = self.interp_i_Ext(t)

        ds = n_dyn.fwd_activity_Chemical(self, v, s, self.compute_Vth(i_Ext))
        dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)/self.neuron_C

        return np.concatenate([dv, ds])

    def forward_network_Jacobian(self, t, y):
        
        """
            (Optional)
            Evaluate the analytical Jacobian of the nervous system ODE system. Can be used to speed up simulations.
            Pararmeters
            
            t : float
                  timepoint in which the right-hand-side function of the ODE is evaluated
                    
            y : NumPy array
                  Vector of current network state (e.g., voltage, synaptic variable, ion channel variables)
        """

        v, s = np.split(y, 2)

        J1, J2 = n_inter.fwd_jacobian12_Linear(self, v, s)
        J3, J4 = n_inter.fwd_jacobian34_Linear(self, v, s, self.compute_Vth(self.interp_i_Ext(t)))

        J_row1 = np.hstack((J1, J2))
        J_row2 = np.hstack((J3, J4))
        J = np.vstack((J_row1, J_row2))

        return J

class CelegansWorm_NervousSystem_Waveforce:
    
    # Identical to tutorial 2 except adding waveforce to the voltage in forward() functions

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self, i_Ext):
        
        return n_inter.compute_vth_Linear(self, i_Ext)

    def forward_Network(self, t, y):

        v, s = y[:self.network_Size], y[self.network_Size:]
        v = v + self.interp_v_Ext(t) # Add waveforce to voltage
        
        i_Linear = n_dyn.fwd_i_Linear(self, v)
        i_Electrical = n_inter.fwd_i_Electrical(self, v)
        i_Chemical = n_inter.fwd_i_Chemical(self, v, s)
        i_Ext = self.interp_i_Ext(t)

        ds = n_dyn.fwd_activity_Chemical(self, v, s, self.compute_Vth(i_Ext))
        dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)/self.neuron_C

        return np.concatenate([dv, ds])

    def forward_network_Jacobian(self, t, y):

        v, s = np.split(y, 2)
        v = v + self.interp_v_Ext(t) # Add waveforce to voltage

        J1, J2 = n_inter.fwd_jacobian12_Linear(self, v, s)
        J3, J4 = n_inter.fwd_jacobian34_Linear(self, v, s, self.compute_Vth(self.interp_i_Ext(t)))

        J_row1 = np.hstack((J1, J2))
        J_row2 = np.hstack((J3, J4))
        J = np.vstack((J_row1, J_row2))

        return J

class CelegansWorm_NervousSystem_PPC:
    
    # Identical to tutorial 2, 3 except 
    # 1. Uses updated connectome by Cook et al, 2019
    # 2. Adds proprioceptive fdb parameters and proprioceptive_FDB() helper function
    # 3. Incorporates proprioceptive fdb input to forward functions

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
        # Proprioceptive feedback parameters (Time delay feedback described in Kim et al, 2024)
        self.fdb_init_Int = int(1.18 * (1/self.timescale))
        self.fdb_delay_Int = int(0.6 * (1/self.timescale))
        
    def compute_Vth(self, i_Ext):
        
        Vth = n_inter.compute_vth_Linear(self, i_Ext)
        
        return Vth
    
    def proprioceptive_FDB(self, NN_mb, v, s, vth, muscle_force, body_state, k):
        
        """
            (REQUIRED IF SIMULATING WITH PROPRIOCEPTIVE FDB)
            Evaluate proprioceptive fdb input given the nervous system, muscle, body states at current timestep k.
            Simulator executes this function at each step of solving Nervous system, muscle and body
            Pararmeters
            
            NN_mb :        Python class
                           Class defined for muscle and body dynamics
                    
            v :            NumPy array of shape (timepoints, self.network_Size)
                           Solution array of network voltage state at timestep k
                    
            s :            NumPy array of shape (timepoints, self.network_Size)
                           Solution array of network synaptic activity state at timestep k
                    
            vth :          NumPy array of shape (timepoints, self.network_Size)
                           Solution array of network resting voltage state at timestep k
                    
            muscle_force : NumPy array of shape (timepoints, muscle dim)
                           Solution array of muscle force at timestep k
                    
            body_state :   NumPy array of shape (timepoints, body dim)
                           Solution array of body state at timestep k
                    
            k :            int
                           Current timestep in integer. i.e., kmin = 0, kmax = simulation duration/timescale 
                    
            Out :          NumPy array of shape (N, )
                           Vector of user-defined proprioceptive feedback input to nervous system
        """
        
        # Here we use the time delay feedback implemented in Kim et al, 2024
        input_FDB = n_inter.fwd_i_ProprioceptiveDelayed(self, NN_mb, v, s, vth, muscle_force, body_state, k)
        
        return input_FDB

    def forward_Network(self, t, y):

        v, s = y[:self.network_Size], y[self.network_Size:]
        v = v + self.input_FDB # Add FDB input to the voltage
        
        i_Linear = n_dyn.fwd_i_Linear(self, v)
        i_Electrical = n_inter.fwd_i_Electrical(self, v)
        i_Chemical = n_inter.fwd_i_Chemical(self, v, s)
        i_Ext = self.interp_i_Ext(t)

        ds = n_dyn.fwd_activity_Chemical(self, v, s, self.compute_Vth(i_Ext))
        dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)/self.neuron_C

        return np.concatenate([dv, ds])

    def forward_network_Jacobian(self, t, y):

        v, s = np.split(y, 2)
        v = v + self.input_FDB # Add FDB input to the voltage

        J1, J2 = n_inter.fwd_jacobian12_Linear(self, v, s)
        J3, J4 = n_inter.fwd_jacobian34_Linear(self, v, s, self.compute_Vth(self.interp_i_Ext(t)))

        J_row1 = np.hstack((J1, J2))
        J_row2 = np.hstack((J3, J4))
        J = np.vstack((J_row1, J_row2))

        return J

class CelegansWorm_NervousSystem_sg:
    
    # Identical to tutorial 4 

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self, i_Ext):
        
        return n_inter.compute_vth_Linear(self, i_Ext)
    
    def proprioceptive_FDB(self, NN_mb, v, s, vth, muscle_force, body_loc, k):
        
        input_FDB = 0
        
        return input_FDB
    
    def translate_Odor(self, env_stim_vec, odor_grad_mat, iext_mat, k): # Defines concentration -> stimulus
        
        """
            (REQUIRED IF SIMULATING WITH SENSORY FDB)
            Evaluates neural stimuli in pA given the envrionmental concentration.
            odor_grad_mat and iext_mat are used if translation requires time derivatives of concentration
            
            Pararmeters
            
            env_stim_vec : NumPy array of shape (self.network_Size, )
                           Odor concentration gradient felt by all neurons at timestep k (n,)
                    
            odor_grad_mat :NumPy array of shape (k-1, self.network_Size)
                           All previous odor concentration up to previous timestep k-1
                    
            iext_mat :     NumPy array of shape (k-1, self.network_Size)
                           All previous final inputs (i_env + i_ext + i_internal) up to previous timestep (k-1, n)
                    
            k :            int
                           Current timestep in integer. i.e., kmin = 0, kmax = simulation duration/timescale 
                    
            Out :          NumPy array of shape (N, )
                           Vector of user-defined neural stimuli translated from odor gradient
        """
        
        # Here we implement a very simple translation where we scale the concentration felt by AWA by 1000pA
        # i.e. 1 concentration -> 1000pA
        
        input_vec = np.zeros(env_stim_vec.shape)
        
        AWA_stim = env_stim_vec[[73, 82]] * 1000
        input_vec[[73, 82]] = AWA_stim

        return input_vec

    def forward_Network(self, t, y):

        v, s = y[:self.network_Size], y[self.network_Size:]
        v = v + self.interp_v_Ext(t)
        
        i_Linear = n_dyn.fwd_i_Linear(self, v)
        i_Electrical = n_inter.fwd_i_Electrical(self, v)
        i_Chemical = n_inter.fwd_i_Chemical(self, v, s)
        i_Ext = self.i_Ext

        ds = n_dyn.fwd_activity_Chemical(self, v, s, self.compute_Vth(i_Ext))
        dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)/self.neuron_C

        return np.concatenate([dv, ds])

    def forward_network_Jacobian(self, t, y):

        v, s = np.split(y, 2)
        v = v + self.interp_v_Ext(t)

        J1, J2 = n_inter.fwd_jacobian12_Linear(self, v, s)
        J3, J4 = n_inter.fwd_jacobian34_Linear(self, v, s, self.compute_Vth(self.i_Ext))

        J_row1 = np.hstack((J1, J2))
        J_row2 = np.hstack((J3, J4))
        J = np.vstack((J_row1, J_row2))

        return J

#----------------------------------------------------------------------------------------------------------------------------------------------------

class CelegansWorm_NervousSystem_Julia:

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self):
        
        Main.eval("""
        
            function compute_Vth(p, i_Ext)
            
                return compute_vth_Linear(p, i_Ext)
                
            end
        
        """)

    def forward_Network(self): # The function is written in Julia to utilize Julia simulation code

        Main.eval("""

            function forward_Network!(du, u, p, t)

                v, s = u[1:p.network_Size], u[p.network_Size+1:end]
                
                i_Linear = fwd_i_Linear(p, v)
                i_Electrical = fwd_i_Electrical(p, v)
                i_Chemical = fwd_i_Chemical(p, v, s)
                i_Ext = p.interp_i_Ext(t)

                ds = fwd_activity_Chemical(p, v, s, compute_Vth(p, i_Ext))
                dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)./p.neuron_C

                du[:] = [dv; ds]

            end

            """)

class CelegansWorm_NervousSystem_Waveforce_Julia:

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self):
        
        Main.eval("""
        
            function compute_Vth(p, i_Ext)
            
                return compute_vth_Linear(p, i_Ext)
                
            end
        
        """)

    def forward_Network(self): # The function is written in Julia to utilize Julia simulation code

        Main.eval("""

            function forward_Network!(du, u, p, t)

                v, s = u[1:p.network_Size], u[p.network_Size+1:end]
                v = v + p.interp_v_Ext(t)
                
                i_Linear = fwd_i_Linear(p, v)
                i_Electrical = fwd_i_Electrical(p, v)
                i_Chemical = fwd_i_Chemical(p, v, s)
                i_Ext = p.interp_i_Ext(t)

                ds = fwd_activity_Chemical(p, v, s, compute_Vth(p, i_Ext))
                dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)./p.neuron_C

                du[:] = [dv; ds]

            end

            """)

class CelegansWorm_NervousSystem_PPC_Julia:

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
        # Proprioceptive feedback 
        self.fdb_init_Int = int(1.18 * (1/self.timescale))
        self.fdb_delay_Int = int(0.6 * (1/self.timescale))
        
    def compute_Vth(self):
        
        Main.eval("""
        
            function compute_Vth(p, i_Ext)
            
                return compute_vth_Linear(p, i_Ext)
                
            end
        
        """)
        
    def proprioceptive_FDB(self):
        
        Main.eval("""
        
            function proprioceptive_FDB(p, NN_mb, v, s, vth, muscle_force, body_loc, k)
            
                return fwd_i_ProprioceptiveDelayed(p, NN_mb, v, s, vth, muscle_force, body_loc, k)
                
            end
        
        """)

    def forward_Network(self): # The function is written in Julia to utilize Julia simulation code

        Main.eval("""

            function forward_Network!(du, u, p, t)

                v, s = u[1:p.network_Size], u[p.network_Size+1:end]
                v = v .+ p.input_FDB
                
                i_Linear = fwd_i_Linear(p, v)
                i_Electrical = fwd_i_Electrical(p, v)
                i_Chemical = fwd_i_Chemical(p, v, s)
                i_Ext = p.interp_i_Ext(t)

                ds = fwd_activity_Chemical(p, v, s, compute_Vth(p, i_Ext))
                dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)./p.neuron_C

                du[:] = [dv; ds]

            end

            """)

class CelegansWorm_NervousSystem_sg_Julia:

    def __init__(self, gap_conn, syn_conn):
        
        # Set network size (>= 1)
        self.network_Size = n_params.CE.n

        # Initialize neurons and their properties
        self.neuron_C = n_dyn.init_neuron_C(capacitance = n_params.CE.cell_caps) # (n,)
        
        self.neuron_Linear = n_dyn.init_neuron_Linear(conductance = n_params.CE.leak_conductances, 
                                                      leak_voltage = n_params.CE.leak_potentials) # (2 x n)
        
        self.neuron_Chemical = n_dyn.init_neuron_Chemical(synaptic_rise_time = n_params.CE.synaptic_rise_tau, 
                                                          synaptic_fall_time = n_params.CE.synaptic_fall_tau, 
                                                          sigmoid_width = n_params.CE.B) # (3 x n)

        # Initialize network and its properties
        self.network_Electrical = n_inter.init_network_Electrical(conn_map = gap_conn, 
                                                                  conductance_map = n_params.CE.gap_conductances, 
                                                                  active_mask = np.ones(279, dtype = 'bool')) # (n x n)

        self.network_Chemical = n_inter.init_network_Chemical(conn_map = syn_conn, 
                                                              conductance_map = n_params.CE.syn_conductances,
                                                              polarity_map = n_params.CE.ei_map, 
                                                              active_mask = np.ones(279, dtype = 'bool')) # (2 x n x n)
        
        # Set initial condition and integration timescale (seconds)
        self.initcond = n_sim.init_Initcond(self)
        self.timescale = 0.01
        
        # Prerequisite for computing Vth (Optional)
        n_inter.init_vth_Linear(self)
        
    def compute_Vth(self):
        
        Main.eval("""
        
            function compute_Vth(p, i_Ext)
            
                return compute_vth_Linear(p, i_Ext)
                
            end
        
        """)
        
    def proprioceptive_FDB(self):
        
        Main.eval("""
        
            function proprioceptive_FDB(p, NN_mb, v, s, vth, muscle_force, body_loc, k)
            
                return zeros(p.network_Size)
                
            end
        
        """)
        
    def translate_Odor(self):

        Main.eval("""

            function translate_Odor(p, env_stim_vec, odor_grad_mat, iext_mat, k)
            
                input_vec = zeros(size(env_stim_vec))
        
                AWA_stim = env_stim_vec[[74, 83]] * 1000
                input_vec[[74, 83]] = AWA_stim

                return input_vec

            end

        """)

    def forward_Network(self): # The function is written in Julia to utilize Julia simulation code

        Main.eval("""

            function forward_Network!(du, u, p, t)

                v, s = u[1:p.network_Size], u[p.network_Size+1:end]
                v = v + p.interp_v_Ext(t)
                
                i_Linear = fwd_i_Linear(p, v)
                i_Electrical = fwd_i_Electrical(p, v)
                i_Chemical = fwd_i_Chemical(p, v, s)
                i_Ext = p.i_Ext

                ds = fwd_activity_Chemical(p, v, s, compute_Vth(p, i_Ext))
                dv = (-(i_Linear + i_Electrical + i_Chemical) + i_Ext)./p.neuron_C

                du[:] = [dv; ds]

            end

            """)
"""
modWorm: Modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans
Copyright (c) 2024-2025 University of Washington. Developed in UW NeuroAI Lab by Jimin Kim.
"""

__author__ = 'Jimin Kim: jk55@u.washington.edu'

import numpy as np

from modWorm import Main

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Python) ########################################################################################################################
#####################################################################################################################################################
# Visco-elastic Rod Model
# McMillen, T., and P. Holmes. "An elastic rod model for anguilliform swimming." Journal of mathematical biology 53.5 (2006): 843-886.

def fwd_body_Coordinates(self, x1, y1, xdot1, ydot1, phi, phidot):

    xvec, yvec, xdot, ydot = np.zeros(self.body_SegCount), np.zeros(self.body_SegCount), np.zeros(self.body_SegCount), np.zeros(self.body_SegCount)
    xvec[0], yvec[0], xdot[0], ydot[0] = x1, y1, xdot1, ydot1

    for j in np.arange(1, len(xvec)):

        j_ = j - 1

        xvec[j] = xvec[j_] + (self.body_SegLength[j] / 2.) * (np.cos(phi[j_]) + np.cos(phi[j]))
        yvec[j] = yvec[j_] + (self.body_SegLength[j] / 2.) * (np.sin(phi[j_]) + np.sin(phi[j]))
        xdot[j] = xdot[j_] - (self.body_SegLength[j] / 2.) * (np.sin(phi[j_]) * phidot[j_] + np.sin(phi[j]) * phidot[j])
        ydot[j] = ydot[j_] + (self.body_SegLength[j] / 2.) * (np.cos(phi[j_]) * phidot[j_] + np.cos(phi[j]) * phidot[j])

    return xvec, yvec, xdot, ydot

def fwd_body_Velocity(self, phi, xdot, ydot):

    v_tan_norm_0 = np.multiply(np.cos(phi), xdot) + np.multiply(np.sin(phi), ydot)
    v_tan_norm_1 = np.multiply(-1 * np.sin(phi), xdot) + np.multiply(np.cos(phi), ydot)

    v_tan_norm = np.concatenate([v_tan_norm_0, v_tan_norm_1])
    v_tan = v_tan_norm[:self.body_SegCount]
    v_norm = v_tan_norm[self.body_SegCount:]

    return v_tan, v_norm

def fwd_body_Curvature(self, fR, fL):

    k = np.divide(4 * (fR - fL) * self.body_Width[:-1], np.subtract(8 * self.body_Stiff[:-1] * self.body_Width[:-1]**2, np.multiply(fR + fL, np.power(self.body_SegLength[:-1], 2))))

    """ curvature scaling """
    k = np.multiply(k, self.body_CurvatureScaling)
    k = k + self.body_CurvatureForcing

    return k

def fwd_body_ContactMoment(self, phi, phidot, k):

    phi_diff = phi[1:] - phi[:-1]
    phi_dot_diff = phidot[1:] - phidot[:-1]

    """ Contact moment """
    M = np.zeros(self.body_SegCount)
    M[:-1] = self.body_StiffMotion[:-1] * (phi_diff - k) + (2 * self.body_RadiusV[:-1]**2 * self.fluid_Damping) * np.divide(phi_dot_diff, self.body_SegLength[:-1])
    Mdiff = np.append(M[0], np.diff(M))

    return Mdiff

def fwd_body_NormalTanForce(self, v_norm, v_tan):

    F_N0 = np.multiply(self.body_RadiusH * self.fluid_Density * self.body_DragCoeff * np.abs(v_norm), v_norm)
    F_N1 = np.multiply(np.sqrt(8 * self.fluid_Density * self.body_RadiusH * self.fluid_Viscosity * np.abs(v_norm)), v_norm)
    F_N = np.add(F_N0, F_N1)

    """ Tan force """
    F_T = np.multiply(2.7 * np.sqrt(2 * self.fluid_Density * self.body_RadiusH * self.fluid_Viscosity * np.abs(v_norm)), v_tan)

    return F_N, F_T

def fwd_body_Forces(self, phi, F_T, F_N):

    """ W computation """
    W0 = np.multiply(-1 * F_T, np.cos(phi)) + np.multiply(F_N, np.sin(phi))
    W1 = np.multiply(-1 * F_T, np.sin(phi)) - np.multiply(F_N, np.cos(phi))
    W = np.concatenate([W0, W1])
    Wx = W[:self.body_SegCount]
    Wy = W[self.body_SegCount:]

    """ W_diff computation """
    Wx_diff0 = np.multiply(np.divide(self.body_SegLength[:-1], self.body_Mass[:-1]), Wx[:-1])
    Wx_diff1 = np.multiply(np.divide(self.body_SegLength[1:], self.body_Mass[1:]), Wx[1:])
    Wx_diff = np.subtract(Wx_diff0, Wx_diff1)
    Wx_diff = np.append(Wx_diff, 0)

    Wy_diff0 = np.multiply(np.divide(self.body_SegLength[:-1], self.body_Mass[:-1]), Wy[:-1])
    Wy_diff1 = np.multiply(np.divide(self.body_SegLength[1:], self.body_Mass[1:]), Wy[1:])
    Wy_diff = np.subtract(Wy_diff0, Wy_diff1)
    Wy_diff = np.append(Wy_diff, 0)

    return Wx, Wy, Wx_diff, Wy_diff

def fwd_body_D2Angles(self, phi, phidot, Mdiff, Wy_diff, Wx_diff):

    Hh_mat = np.diag(self.body_SegLength) + np.diag(self.body_SegLength[1:], 1)
    Gh_mat = np.diag(self.body_SegLength) + np.diag(self.body_SegLength[1:], -1)
    Gcos = np.multiply(Gh_mat / 2., Gmat(np.cos(phi)))
    Gsin = np.multiply(Gh_mat / 2., Gmat(np.sin(phi)))
    Hcos = np.multiply(Hh_mat / 2., Hmat(np.cos(phi)))
    Hsin = np.multiply(Hh_mat / 2., Hmat(np.sin(phi)))

    phi_ddot_numerator0 = np.dot(np.dot(np.dot(-Gcos, self.body_ForceTransform), Hsin) + np.dot(np.dot(Gsin, self.body_ForceTransform), Hcos), np.power(phidot, 2))
    phi_ddot_numerator1 = Mdiff + np.dot(np.dot(Gcos, self.body_ForceTransform), Wy_diff) - np.dot(np.dot(Gsin, self.body_ForceTransform), Wx_diff)
    phi_ddot_numerator = np.add(phi_ddot_numerator0, phi_ddot_numerator1)

    phi_ddot_denominator = self.body_InertiaSeg - np.dot(np.dot(Gcos, self.body_ForceTransform), Hcos) - np.dot(np.dot(Gsin, self.body_ForceTransform), Hsin)
    phi_ddot = np.linalg.solve(phi_ddot_denominator, phi_ddot_numerator)

    return phi_ddot

def fwd_body_D2Coordinates(self, phi, phidot, phi_ddot, Wx, Wy):

    C_ddot = np.zeros(self.body_SegCount)
    cs1 = np.cos(phi[0]) * phidot[0]**2 + np.sin(phi[0]) * phi_ddot[0]
    cs2 = np.cos(phi[1]) * phidot[1]**2 + np.sin(phi[1]) * phi_ddot[1]
    C_ddot[1] = -self.body_SegLength[1] / 2. * (cs1 + cs2)

    S_ddot = np.zeros(self.body_SegCount)
    sc1 = np.sin(phi[0]) * phidot[0]**2 - np.cos(phi[0]) * phi_ddot[0]
    sc2 = np.sin(phi[1]) * phidot[1]**2 - np.cos(phi[1]) * phi_ddot[1]
    S_ddot[1] = -self.body_SegLength[1] / 2. * (sc1 + sc2)

    for j in range(2, self.body_SegCount):

        cs_i = np.cos(phi[j]) * phidot[j]**2 + np.sin(phi[j]) * phi_ddot[j]
        cs_im1 = np.cos(phi[j-1]) * phidot[j-1]**2 + np.sin(phi[j-1]) * phi_ddot[j-1]

        sc_i = np.sin(phi[j]) * phidot[j]**2 - np.cos(phi[j]) * phi_ddot[j]
        sc_im1 = np.sin(phi[j-1]) * phidot[j-1]**2 - np.cos(phi[j-1]) * phi_ddot[j-1]

        C_ddot[j] = C_ddot[j-1] - self.body_SegLength[j] / 2. * cs_im1 - self.body_SegLength[j] / 2. * cs_i
        S_ddot[j] = S_ddot[j-1] - self.body_SegLength[j] / 2. * sc_im1 - self.body_SegLength[j] / 2. * sc_i

    m_sum = np.sum(self.body_Mass)

    x_ddot_1 = np.reciprocal(m_sum) * np.sum(np.multiply(self.body_SegLength, Wx) - np.multiply(self.body_Mass, C_ddot))
    y_ddot_1 = np.reciprocal(m_sum) * np.sum(np.multiply(self.body_SegLength, Wy) - np.multiply(self.body_Mass, S_ddot))

    return x_ddot_1, y_ddot_1

def Gmat(phi):

    G = np.diag(phi) + np.diag(phi[1:], -1)

    return G

def Hmat(phi):

    H = np.diag(np.append(phi[:-1], 0)) + np.diag(phi[1:], 1)

    return H

#####################################################################################################################################################
# COMPUTE FUNCTIONS (Julia) #########################################################################################################################
#####################################################################################################################################################
# Visco-elastic Rod Model
# McMillen, T., and P. Holmes. "An elastic rod model for anguilliform swimming." Journal of mathematical biology 53.5 (2006): 843-886.

Main.eval("""

    function fwd_body_Coordinates(p, x1, y1, xdot1, ydot1, phi, phidot)

        xvec, yvec, xdot, ydot = zeros(p.body_SegCount), zeros(p.body_SegCount), zeros(p.body_SegCount), zeros(p.body_SegCount)
        xvec[1], yvec[1], xdot[1], ydot[1] = x1, y1, xdot1, ydot1

        for j = 2:size(xvec)[1]

            j_ = j - 1

            xvec[j] = xvec[j_] + (p.body_SegLength[j] / 2) * (cos(phi[j_]) + cos(phi[j]))
            yvec[j] = yvec[j_] + (p.body_SegLength[j] / 2) * (sin(phi[j_]) + sin(phi[j]))
            xdot[j] = xdot[j_] - (p.body_SegLength[j] / 2) * (sin(phi[j_]) * phidot[j_] + sin(phi[j]) * phidot[j])
            ydot[j] = ydot[j_] + (p.body_SegLength[j] / 2) * (cos(phi[j_]) * phidot[j_] + cos(phi[j]) * phidot[j])

        end

        return xvec, yvec, xdot, ydot

    end

    function fwd_body_Velocity(p, phi, xdot, ydot)

        v_tan_norm_0 = (cos.(phi) .* xdot) + (sin.(phi) .* ydot)
        v_tan_norm_1 = (-1 * sin.(phi) .* xdot) + (cos.(phi) .* ydot)

        v_tan_norm = [v_tan_norm_0 ; v_tan_norm_1]
        v_tan = v_tan_norm[1:p.body_SegCount]
        v_norm = v_tan_norm[p.body_SegCount+1:end]

        return v_tan, v_norm

    end

    function fwd_body_Curvature(p, fR, fL)

        k_numerator = 4 * (fR - fL) .* p.body_Width[1:end-1]
        k_denominator = (8 * p.body_Stiff[1:end-1] .* p.body_Width[1:end-1].^2) - ((fR + fL) .* p.body_SegLength[1:end-1].^2)

        k = k_numerator ./ k_denominator
        k = k * p.body_CurvatureScaling
        k = k .+ p.body_CurvatureForcing

        return k

    end

    function fwd_body_ContactMoment(p, phi, phidot, k)

        phi_diff = phi[2:end] - phi[1:end-1]
        phi_dot_diff = phidot[2:end] - phidot[1:end-1]

        M = zeros(p.body_SegCount)
        M[1:end-1] = p.body_StiffMotion[1:end-1] .* (phi_diff - k) + (2 * p.body_RadiusV[1:end-1].^2 * p.fluid_Damping) .* (phi_dot_diff ./ p.body_SegLength[1:end-1])
        Mdiff = [M[1] ; diff(M)]

        return Mdiff

    end

    function fwd_body_NormalTanForce(p, v_norm, v_tan)

        F_N0 = (p.body_RadiusH * p.fluid_Density * p.body_DragCoeff .* abs.(v_norm)) .* v_norm
        F_N1 = (sqrt.(8 * p.fluid_Density .* p.body_RadiusH .* p.fluid_Viscosity .* abs.(v_norm))) .* v_norm
        F_N = F_N0 + F_N1

        F_T = (2.7 * sqrt.(2 * p.fluid_Density .* p.body_RadiusH .* p.fluid_Viscosity .* abs.(v_norm))) .* v_tan

        return F_N, F_T

    end

    function fwd_body_Forces(p, phi, F_T, F_N)

        W0 = (-1 * F_T .* cos.(phi)) + (F_N .* sin.(phi))
        W1 = (-1 * F_T .* sin.(phi)) - (F_N .* cos.(phi))
        W = [W0 ; W1]
        Wx = W[1:p.body_SegCount]
        Wy = W[p.body_SegCount+1:end]

        Wx_diff0 = (p.body_SegLength[1:end-1] ./ p.body_Mass[1:end-1]) .* Wx[1:end-1]
        Wx_diff1 = (p.body_SegLength[2:end] ./ p.body_Mass[2:end]) .* Wx[2:end]
        Wx_diff = Wx_diff0 - Wx_diff1
        Wx_diff = [Wx_diff ; 0]

        Wy_diff0 = (p.body_SegLength[1:end-1] ./ p.body_Mass[1:end-1]) .* Wy[1:end-1]
        Wy_diff1 = (p.body_SegLength[2:end] ./ p.body_Mass[2:end]) .* Wy[2:end]
        Wy_diff = Wy_diff0 - Wy_diff1
        Wy_diff = [Wy_diff ; 0]

        return Wx, Wy, Wx_diff, Wy_diff

    end

    function fwd_body_D2Angles(p, phi, phidot, Mdiff, Wy_diff, Wx_diff)

        Hh_mat = Bidiagonal(p.body_SegLength, p.body_SegLength[2:end], :U)
        Gh_mat = Bidiagonal(p.body_SegLength, p.body_SegLength[2:end], :L)

        Gcos = (Gh_mat ./ 2) .* Gmat(cos.(phi))
        Gsin = (Gh_mat ./ 2) .* Gmat(sin.(phi))
        Hcos = (Hh_mat ./ 2) .* Hmat(cos.(phi))
        Hsin = (Hh_mat ./ 2) .* Hmat(sin.(phi))

        phi_ddot_numerator0 = ((-Gcos * p.body_ForceTransform) * Hsin + (Gsin * p.body_ForceTransform) * Hcos) * phidot.^2
        phi_ddot_numerator1 = Mdiff + ((Gcos * p.body_ForceTransform) * Wy_diff) - ((Gsin * p.body_ForceTransform) * Wx_diff)
        phi_ddot_numerator = phi_ddot_numerator0 + phi_ddot_numerator1

        phi_ddot_denominator = p.body_InertiaSeg - ((Gcos * p.body_ForceTransform) * Hcos) - ((Gsin * p.body_ForceTransform) * Hsin)
        phi_ddot = phi_ddot_denominator \ phi_ddot_numerator

        return phi_ddot

    end

    function fwd_body_D2Coordinates(p, phi, phidot, phi_ddot, Wx, Wy)

        C_ddot = zeros(p.body_SegCount)
        cs1 = cos(phi[1]) * phidot[1]^2 + sin(phi[1]) * phi_ddot[1]
        cs2 = cos(phi[2]) * phidot[2]^2 + sin(phi[2]) * phi_ddot[2]
        C_ddot[2] = -p.body_SegLength[2] / 2 * (cs1 + cs2)

        S_ddot = zeros(p.body_SegCount)
        sc1 = sin(phi[1]) * phidot[1]^2 - cos(phi[1]) * phi_ddot[1]
        sc2 = sin(phi[2]) * phidot[2]^2 - cos(phi[2]) * phi_ddot[2]
        S_ddot[2] = -p.body_SegLength[2] / 2 * (sc1 + sc2)

        for j = 3:p.body_SegCount

            cs_i = cos(phi[j]) * phidot[j]^2 + sin(phi[j]) * phi_ddot[j]
            cs_im1 = cos(phi[j-1]) * phidot[j-1]^2 + sin(phi[j-1]) * phi_ddot[j-1]

            sc_i = sin(phi[j]) * phidot[j]^2 - cos(phi[j]) * phi_ddot[j]
            sc_im1 = sin(phi[j-1]) * phidot[j-1]^2 - cos(phi[j-1]) * phi_ddot[j-1]

            C_ddot[j] = C_ddot[j-1] - p.body_SegLength[j] / 2 * cs_im1 - p.body_SegLength[j] / 2 * cs_i
            S_ddot[j] = S_ddot[j-1] - p.body_SegLength[j] / 2 * sc_im1 - p.body_SegLength[j] / 2 * sc_i

        end

        m_sum = sum(p.body_Mass)

        x_ddot_1 = (1 / m_sum) * sum((p.body_SegLength .* Wx) - (p.body_Mass .* C_ddot))
        y_ddot_1 = (1 / m_sum) * sum((p.body_SegLength .* Wy) - (p.body_Mass .* S_ddot))

        return x_ddot_1, y_ddot_1

    end

    function Gmat(phi)

        G = Bidiagonal(phi, phi[2:end], :L)

        return G

    end

    function Hmat(phi)

        H = Bidiagonal([phi[1:end-1] ; 0], phi[2:end], :U)

        return H

    end

    """)

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 04 12:57:00 2021

@author: Leonardo
"""

import CoolProp.CoolProp as CP
import CoolProp
import numpy as np
import time
from line_profiler import LineProfiler
from collections import namedtuple


import CoolProp.CoolProp as CP
import CoolProp
import numpy as np
import time
from line_profiler import LineProfiler
from collections import namedtuple
from numba import jit, njit, vectorize, int64


def h_bifasico(G_f, fluido, d_i_tubo, P_f_in, x_f_in, q_h, HEOS_fluido, BICUBIC_fluido):
   
    # Kim e Mudawar
    Props_vapor = BICUBIC_fluido
    Props_vapor.update(CoolProp.PQ_INPUTS, P_f_in, 1)
    rho_g = Props_vapor.rhomass()
    mu_g = Props_vapor.viscosity()
    P_crit = Props_vapor.p_critical()
    i_g = Props_vapor.hmass()
        
    Props_liquido = BICUBIC_fluido
    Props_liquido.update(CoolProp.PQ_INPUTS, P_f_in, 0)   
    rho_l = Props_liquido.rhomass()
    mu_l = Props_liquido.viscosity()
    k_l = Props_liquido.conductivity()
    Pr_l = Props_liquido.Prandtl()
    i_l = Props_liquido.hmass()
    
    sigma = surface_tension_global
    h_fg = i_g - i_l
      
    # # Números adimensionais 
    P_r = P_f_in / P_crit
    Re_l = G_f * (1 - x_f_in) * d_i_tubo / mu_l
    We_fo = G_f ** 2 * d_i_tubo / (rho_l * sigma)
    X_tt = (mu_l / mu_g) ** 0.1 * ((1 - x_f_in) / x_f_in) ** 0.9 * (rho_g / rho_l) ** 0.5
    Bo = q_h / (G_f * h_fg)
    
    # Calculo dos coeficientes de transcal
    h_dittus = 0.023 * Re_l ** 0.8 * Pr_l ** 0.4 * k_l / d_i_tubo
    h_cb = (5.2 * Bo ** 0.08 * We_fo ** (-0.54) + 3.5 * (1 / X_tt) ** 0.94 * (rho_g / rho_l) ** 0.25) * h_dittus
    h_nb = (2345 * Bo ** 0.7 * P_r ** 0.38 * (1 - x_f_in) ** (-0.51)) * h_dittus
    h_tp = (h_cb ** 2 + h_nb ** 2) ** 0.5
   
    # Oh and Son 
    h_tp1 = 0.034 * Re_l ** 0.8 * Pr_l ** 0.3 * 1.58 * (1 / X_tt) ** 0.87 * (k_l / d_i_tubo)
   
    # Li and Wu
    Bd = 9.81 * (rho_l - rho_g) * d_i_tubo ** 2 / sigma
    h_tp2 = 334 * Bo ** 0.3 * (Bd * Re_l ** 0.36) ** 0.4 * k_l / d_i_tubo
    
   
    return h_tp


def h_monofasico(G_f, fluido, d_i_tubo, P_f_in, i_f_in, BICUBIC_fluido):
    
    Props = BICUBIC_fluido
    Props.update(CoolProp.HmassP_INPUTS, i_f_in, P_f_in)
    
    k_f = Props.conductivity()
    Pr_f = Props.Prandtl()
    mu_f = Props.viscosity()
    Re_d_f = G_f * d_i_tubo / mu_f

    "Petukhov"
    f = (1.8 * np.log10(Re_d_f) - 1.5) ** (-2)
    h_f = (k_f / d_i_tubo) * (f / 8) * (Re_d_f) * Pr_f / (1.07 + 12.7 * (f/8)**(1/2) * (Pr_f**(2/3) - 1))
    # * (1+ (d_i_tubo/L_tubo)**(2/3))

    "Dittus Boelter"
    #h_f = (k_f / d_i_tubo) * 0.0265 * (Re_d_f**(4/5)) * (Pr_f)**(0.3)

    "Webb"
    # f = (1.8 * np.log10(Re_d_f) - 1.5) ** (-2)
    # h_f = (k_f / d_i_tubo) * (f / 8) * (Re_d_f) * Pr_f / (1.07 + 9 * (f/8)**(1/2) * (Pr_f - 1) * Pr_f ** (-1/4)) * (1+ (d_i_tubo/L_tubo)**(2/3))

    return h_f


def calor_monofasico(m_ar, m_f, cp_ar, cp_f_in, UA, T_ar_in, T_f_in):

    C_ar = (m_ar * cp_ar)
    C_f = (m_f * cp_f_in)
    C_min = min(C_ar, C_f)
    C_max = max(C_ar, C_f)
    C_r = C_min / C_max
    NTU = UA / C_min

    efi_hx = 1 - np.exp((NTU**0.22 / C_r) * (np.exp(-C_r * NTU**0.78) - 1))
    Q_max = C_min * abs(T_ar_in - T_f_in)
    Q = efi_hx * Q_max

    return Q


def serpentina(Passes, N_elementos, Q, i_f_o, i_f_i, P_f_out, P_f_in, DP, m_f, cp_ar, m_ar, T_ar_in, T_ar_out, DP_ar, P_ar_in, P_ar_out, N_v):
    # Faz o calculo das novas propriedades do fluido

    posição = 0
    for passe, tubos in enumerate(Passes):  # Análise para quantidade de tubo
    
        for e in range(N_elementos):
            
            i_f_o[posição:posição + tubos, e] = i_f_i[posição:posição + tubos, e] + Q[posição:posição + tubos, e] / m_f[posição:posição + tubos, e]
            P_f_out[posição:posição + tubos, e] = P_f_in[posição:posição + tubos, e] - DP[posição:posição + tubos, e]
       
            if not e == N_elementos - 1:
                i_f_i[posição:posição + tubos, e + 1] = i_f_o[posição:posição + tubos, e]
                P_f_in[posição:posição + tubos, e + 1] = P_f_out[posição:posição + tubos, e] 
                
        i_saida = np.mean([i_f_o[posição:posição + tubos, -1]])  # entalpia média de saída do passe
        P_saida = np.mean([P_f_out[posição:posição + tubos, -1]])
        posição += tubos

        if passe + 1 != len(Passes):
            i_f_i[posição:posição + Passes[passe + 1], 0] = i_saida
            P_f_in[posição:posição + Passes[passe + 1], 0] = P_saida

    T_ar_out = T_ar_in - Q / (m_ar * cp_ar)
    P_ar_out = P_ar_in - DP_ar

    T_ar_medio = (T_ar_in + T_ar_out) / 2
    
    return i_f_o, i_f_i, P_f_in, P_f_out, T_ar_in, T_ar_out, T_ar_medio


def propriedades_fluido(P_f_in, P_f_out, i_f_in, i_f_out, Q, m_f, N_v, N_elementos, BICUBIC_fluido):
    
    x_f_in = np.empty((N_v, N_elementos))
    x_f_out = np.empty((N_v, N_elementos))
    T_f_in = np.empty((N_v, N_elementos))
    T_f_out = np.empty((N_v, N_elementos))
    cp_f_in = np.empty((N_v, N_elementos))
    cp_f_out = np.empty((N_v, N_elementos))
    Q_vapor = np.empty((N_v, N_elementos))
    Q_liquido = np.empty((N_v, N_elementos))
    fracao_vapor = np.empty((N_v, N_elementos))
    fracao_bifasico = np.empty((N_v, N_elementos))
    
    Props = BICUBIC_fluido
    
    for i in range(N_v):
        
        for e in range(N_elementos):
            Props.update(CoolProp.HmassP_INPUTS, i_f_in[i][e], P_f_in[i][e])
            x_f_in[i][e] = Props.Q()
            T_f_in[i][e] = Props.T()
            cp_f_in[i][e] = Props.cpmass()
            
            Props.update(CoolProp.HmassP_INPUTS, i_f_out[i][e], P_f_out[i][e])
            x_f_out[i][e] = Props.Q()
            T_f_out[i][e] = Props.T()
            cp_f_out[i][e] = Props.cpmass()
            
            Props.update(CoolProp.PQ_INPUTS, P_f_in[i][e], 1)
            i_vapor = Props.hmass()
            
            Props.update(CoolProp.PQ_INPUTS, P_f_in[i][e], 0)
            i_liquido = Props.hmass()
                                  
            if x_f_in[i][e] <= -1 and i_f_in[i][e] > i_vapor:
                x_f_in[i][e] = 1
            elif x_f_in[i][e] <= -1 and i_f_in[i][e] < i_liquido:
                x_f_in[i][e] = 0
                                
            if x_f_out[i][e] <= -1 and i_f_out[i][e] > i_vapor:
                x_f_out[i][e] = 1
            elif x_f_out[i][e] <= -1 and i_f_out[i][e] < i_liquido:
                x_f_out[i][e] = 0
            
            if x_f_in[i][e] >= 1 or x_f_in[i][e] <= 0:
                cp_f_in[i][e] = cp_f_in[i][e]
            else:
                cp_f_in[i][e] = 10000000000
                
            if x_f_out[i][e] >= 1 or x_f_out[i][e] <= 0:
                cp_f_out[i][e] = cp_f_out[i][e]
            else:
                cp_f_out[i][e] = 10000000000
                            
            Q_vapor[i][e] = - m_f[i][e] * (i_f_in[i][e] - i_vapor)
            Q_liquido[i][e] = - m_f[i][e] * (i_f_in[i][e] - i_liquido)
        
            # verifica qual fração de vapor e bifasico
            if Q_vapor[i][e] > 0 and Q_vapor[i][e] > Q[i][e]:
        
                fracao_vapor[i][e] = 1
                fracao_bifasico[i][e] = 0
        
            elif Q_vapor[i][e] > 0 and Q_vapor[i][e] < Q[i][e]:
        
                fracao_vapor[i][e] = Q_vapor[i][e] / Q[i][e]
                fracao_bifasico[i][e] = 1 - fracao_vapor[i][e]
        
            elif Q_liquido[i][e] > 0 and Q_liquido[i][e] > Q[i][e]:
        
                fracao_bifasico[i][e] = 1
                fracao_vapor[i][e] = 0
        
            elif Q_liquido[i][e] > 0 and Q_liquido[i][e] < Q[i][e]:
        
                fracao_bifasico[i][e] = Q_liquido[i][e] / Q[i][e]
                fracao_vapor[i][e] = 0
        
            else:
        
                fracao_bifasico[i][e] = 0
                fracao_vapor[i][e] = 0

    return x_f_in, x_f_out, cp_f_in, cp_f_out, T_f_in, T_f_out, fracao_vapor, fracao_bifasico

def propriedades_fluido_vec(P_f_in, P_f_out, i_f_in, i_f_out, Q, m_f, BICUBIC_fluido):
    

    Props = BICUBIC_fluido
    
    Props.update(CoolProp.HmassP_INPUTS, i_f_in, P_f_in)
    x_f_in = Props.Q()
    T_f_in= Props.T()
    cp_f_in = Props.cpmass()
    
    Props.update(CoolProp.HmassP_INPUTS, i_f_out, P_f_out)
    x_f_out = Props.Q()
    T_f_out = Props.T()
    cp_f_out = Props.cpmass()
    
    Props.update(CoolProp.PQ_INPUTS, P_f_in, 1.0)
    i_vapor_in = Props.hmass()
    
    Props.update(CoolProp.PQ_INPUTS, P_f_in, 0.0)
    i_liquido_in = Props.hmass()
    
    Props.update(CoolProp.PQ_INPUTS, P_f_out, 1.0)
    i_vapor_out = Props.hmass()
    
    Props.update(CoolProp.PQ_INPUTS, P_f_out, 0.0)
    i_liquido_out = Props.hmass()   
                   
    if x_f_in <= -1.0 and i_f_in > i_vapor_in:
        x_f_in = 1.0
    elif x_f_in <= -1.0 and i_f_in < i_liquido_in:
        x_f_in = 0.0
    else:
        x_f_in = x_f_in
                        
    if x_f_out <= -1.0 and i_f_out > i_vapor_out:
        x_f_out = 1.0
    elif x_f_out <= -1.0 and i_f_out < i_liquido_out:
        x_f_out = 0.0
    else:
        x_f_out = x_f_out
    
    if x_f_in >= 1.0 or x_f_in <= 0.0:
        cp_f_in = cp_f_in
    else:
        cp_f_in = 10000000000.0
        
    if x_f_out >= 1.0 or x_f_out <= 0.0:
        cp_f_out = cp_f_out
    else:
        cp_f_out = 10000000000.0
                    
    Q_vapor = - m_f * (i_f_in - i_vapor_out)
    Q_liquido = - m_f * (i_f_in - i_liquido_out)
    
    # verifica qual fração de vapor e bifasico
   
    if Q_liquido > 0.0 and Q_liquido > Q:

        fracao_bifasico = 0.0
        fracao_vapor = 0.0

    elif Q_liquido > 0.0 and Q_liquido < Q:

        fracao_bifasico = 1.0 - Q_liquido / Q
        fracao_vapor = 0.0
        
    if Q_vapor > 0.0 and Q_vapor > Q:

        fracao_vapor = 0.0
        fracao_bifasico = 1.0

    elif Q_vapor > 0.0 and Q_vapor < Q:
        
        fracao_bifasico = Q_vapor / Q
        fracao_vapor = 1 - fracao_bifasico

    else:

        fracao_bifasico = 0.0
        fracao_vapor = 1

    return x_f_in, x_f_out, cp_f_in, cp_f_out, T_f_in, T_f_out, fracao_vapor, fracao_bifasico


def propriedades_ar(P_ar_in, P_ar_out, T_ar_in, T_ar_out, N_v, N_elementos, HEOS_Ar):
    
    rho_ar_in = np.empty((N_v, N_elementos))
    rho_ar_out = np.empty((N_v, N_elementos))
    cp_ar_in = np.empty((N_v, N_elementos))
    cp_ar_out = np.empty((N_v, N_elementos))
    mu_ar_in = np.empty((N_v, N_elementos))
    mu_ar_out = np.empty((N_v, N_elementos))
    Pr_ar_in = np.empty((N_v, N_elementos))
    Pr_ar_out = np.empty((N_v, N_elementos))
    
    Props_ar = HEOS_Ar
    
    for i in range(N_v):
        
        for e in range(N_elementos):
            Props_ar.update(CoolProp.PT_INPUTS, P_ar_in[i][e], T_ar_in[i][e])
            rho_ar_in[i][e] = Props_ar.rhomass()
            cp_ar_in[i][e] = Props_ar.cpmass()
            mu_ar_in[i][e] = Props_ar.viscosity()
            Pr_ar_in[i][e] = Props_ar.Prandtl()
            
            Props_ar.update(CoolProp.PT_INPUTS, P_ar_out[i][e], T_ar_out[i][e])           
            rho_ar_out[i][e] = Props_ar.rhomass()
            cp_ar_out[i][e] = Props_ar.cpmass()
            mu_ar_out[i][e] = Props_ar.viscosity()
            Pr_ar_out[i][e] = Props_ar.Prandtl()
            
    return rho_ar_in, rho_ar_out, cp_ar_in, cp_ar_out, mu_ar_in, mu_ar_out, Pr_ar_in, Pr_ar_out


def propriedades_ar_vec(P_ar_in, P_ar_out, T_ar_in, T_ar_out, HEOS_Ar):
    
    Props_ar = HEOS_Ar
    
    Props_ar.update(CoolProp.PT_INPUTS, P_ar_in, T_ar_in)
    rho_ar_in = Props_ar.rhomass()
    cp_ar_in = Props_ar.cpmass()
    mu_ar_in = Props_ar.viscosity()
    Pr_ar_in = Props_ar.Prandtl()
                       
    Props_ar.update(CoolProp.PT_INPUTS, P_ar_out, T_ar_out)           
    rho_ar_out = Props_ar.rhomass()
    cp_ar_out = Props_ar.cpmass()
    mu_ar_out = Props_ar.viscosity()
    Pr_ar_out = Props_ar.Prandtl()
            

    return rho_ar_in, rho_ar_out, cp_ar_in, cp_ar_out, mu_ar_in, mu_ar_out, Pr_ar_in, Pr_ar_out


def fator_atrito(fluido, P_f_in, i_f_in, mu_f_in, G_f, d_i_tubo, rug):

    Re_d_f = G_f * d_i_tubo / mu_f_in
    A = (2.457 * np.log(1 / (((7 / Re_d_f) ** 0.9) + 0.27 * (rug / d_i_tubo)))) ** 16
    B = (37530 / Re_d_f)**16
    f_d = 8 * ((8 / Re_d_f) ** 12 + (A + B) ** (-3 / 2)) ** (1 / 12)

    return f_d


def DP_monofasico(fluido, P_f_in, i_f_in, G_f, d_i_tubo, L_tubo, rug, BICUBIC_fluido):

    Props = BICUBIC_fluido
    Props.update(CoolProp.HmassP_INPUTS, i_f_in, P_f_in)
    rho = Props.rhomass()
    mu_f_in = Props.viscosity()
    f = fator_atrito(fluido, P_f_in, i_f_in, mu_f_in, G_f, d_i_tubo, rug)
    DP = (f * (G_f**2) / (2 * rho * d_i_tubo)) * L_tubo

    return DP


def DP_bifasico(fluido, P_f_in, i_f_in, x_f_in, G_f, d_i_tubo, L_tubo, q_h, HEOS_fluido, BICUBIC_fluido):
    
    # Müller-Steinhagen and Heck
    # rho_g = CP.PropsSI('D', 'P', P_f_in, 'Q', 1, fluido)
    # rho_l = CP.PropsSI('D', 'P', P_f_in, 'Q', 0, fluido)

    # mu_g = CP.PropsSI('VISCOSITY', 'P', P_f_in, 'Q', 1, fluido)
    # mu_l = CP.PropsSI('VISCOSITY', 'P', P_f_in, 'Q', 0, fluido)
    # Re_go = G_f * d_i_tubo / mu_g
    # Re_lo = G_f * d_i_tubo / mu_l
    # f_go = 0.316 * Re_go ** (-1/4)
    # f_lo = 0.316 * Re_lo ** (-1/4)
   
    # dpdz_lo = f_lo * (G_f**2) / (2 * rho_l * d_i_tubo)
    # dpdz_go = f_go * (G_f**2) / (2 * rho_g * d_i_tubo)

    # DP = ((dpdz_lo + 2 * (dpdz_go - dpdz_lo) * x_f_in) * ((1 - x_f_in)**(1/3)) + dpdz_go * (x_f_in**3)) * L_tubo

    # Kim e Mudawar
    Props_fluido = BICUBIC_fluido
    
    Props_fluido.update(CoolProp.PQ_INPUTS, P_f_in, 1)
    rho_g = Props_fluido.rhomass()
    mu_g = Props_fluido.viscosity()
    i_g = Props_fluido.hmass()
      
    Props_fluido.update(CoolProp.PQ_INPUTS, P_f_in, 0)
    rho_l = Props_fluido.rhomass()
    mu_l = Props_fluido.viscosity()
    i_l = Props_fluido.hmass()
   
    sigma = surface_tension_global
    h_fg = i_g - i_l
    
    Re_l = G_f * (1.0 - x_f_in) * d_i_tubo / mu_l
    Re_g = G_f * x_f_in * d_i_tubo / mu_g
    Re_lo = G_f * d_i_tubo / mu_l
    Su_go = rho_g * sigma * d_i_tubo / (mu_g ** 2)
    We_fo = G_f ** 2 * d_i_tubo / (rho_l * sigma)
    Bo = q_h / (G_f * h_fg)
       
    # Valores para C
    if Re_l >= 2_000 and Re_g >= 2_000:  # tt
        C = 0.39 * Re_lo ** 0.03 * Su_go ** 0.1 * (rho_l / rho_g) ** 0.35
    elif Re_l >= 2_000 and Re_g < 2_000:  # tv
        C = 8.7e-4 * Re_lo ** 0.17 * Su_go ** 0.5 * (rho_l / rho_g) ** 0.14
    elif Re_l < 2_000 and Re_g >= 2_000:  # vt
        C = 0.0015 * Re_lo ** 0.59 * Su_go ** 0.19 * (rho_l / rho_g) ** 0.36
    elif Re_l < 2_000 and Re_g < 2_000:  # vv
        C = 3.5e-5 * Re_lo ** 0.44 * Su_go ** 0.5 * (rho_l / rho_g) ** 0.48
          
    # Valores corrigidos para C
    if Re_l >= 2000:
        C = C * (1 + 60 * We_fo ** 0.32 * Bo ** 0.78)
    elif Re_l < 2000:
        C = C * (1 + 530 * We_fo ** 0.52 * Bo ** 1.09)
    
    # Valores para o fator de atrito do liquido
    if Re_l < 2_000:
        f_l = 16 * Re_l ** (-1)
    elif 2_000  <= Re_l < 20_000:
        f_l = 0.079 * Re_l ** (-0.25)
    elif Re_l >= 20_000:
        f_l = 0.046 * Re_l ** (-0.2)
       
    # Valores para o fator de atrito do vapor
    if Re_g < 2_000:
        f_g = 16 * Re_g ** (-1)
    elif 2_000  <= Re_g < 20_000:
        f_g = 0.079 * Re_g ** (-0.25)
    elif Re_g >= 20_000:
        f_g = 0.046 * Re_g ** (-0.2)  
      
    # Diferenciais de pressão vapor e liquido
    dpdz_l = - 2 * f_l * G_f ** 2 * (1 - x_f_in) ** 2 / (rho_l * d_i_tubo) 
    dpdz_g = - 2 * f_g * G_f ** 2 * x_f_in ** 2 / (rho_g * d_i_tubo) 
    X = (dpdz_l / dpdz_g) ** (1/2)
    phi_l = (1 + C / X + 1 / (X ** 2)) ** (1/2)
    
    # Queda de pressão total
    DP_kim = - dpdz_l * phi_l **2 * L_tubo

    return DP_kim


def fatores(V_ar_core, rho_ar_in, mu_ar_in, D_h):

    # fator de atrito
    # Re_lp = V_ar_core * rho_ar_in * L_p / mu_ar_in

    # if (Re_lp >= 230):

    #     f1 = 4.97 * (Re_lp ** (0.6049 - 1.064 / (theta ** 0.2))) * ((np.log((F_t / F_p) ** 0.5 + 0.9)) ** (-0.527))
    #     f2 = (((D_h / L_p) * np.log(0.3 * Re_lp)) ** (-2.966)) * ((F_p / L_l) ** (-0.7931 * T_p / T_h))
    #     f3 = ((T_p / D_m) ** (-0.0446)) * ((np.log(1.2 + (L_p / F_p) ** 1.4)) ** (-3.553)) * (theta ** (-0.477))
    #     f_ar = f1 * f2 * f3

    # elif (Re_lp <= 130):

    #     f1 = 14.39 * (Re_lp ** (-0.805 * F_p / F_l)) * ((np.log(1 + F_p / L_p)) ** 3.04)
    #     f2 = ((np.log(((F_t / F_p) ** 0.48) + 0.9)) ** (-1.435)) * ((D_h / L_p) ** (-3.01)) * ((np.log(0.5 * Re_lp)) ** (-3.01))
    #     f3 = ((F_p / L_l) ** (-0.308)) * ((F_d / L_l) ** (-0.308)) * (np.exp(-0.1167 * T_p / D_m)) * (theta ** 0.35)
    #     f_ar = f1 * f2 * f3

    # else:  # Faixa de descontinuidade

    #     w = 3.6 - 0.02 * Re_lp

    #     # f_130
    #     Re_lp = 130
    #     f1 = 14.39 * (Re_lp ** (-0.805 * F_p / F_l)) * ((np.log(1 + F_p / L_p)) ** 3.04)
    #     f2 = ((np.log(((F_t / F_p) ** 0.48) + 0.9)) ** (-1.435)) * ((D_h / L_p) ** (-3.01)) * ((np.log(0.5 * Re_lp)) ** (-3.01))
    #     f3 = ((F_p / L_l) ** (-0.308)) * ((F_d / L_l) ** (-0.308)) * (np.exp(-0.1167 * T_p / D_m)) * (theta ** 0.35)
    #     f_130 = f1 * f2 * f3

    #     # f_230
    #     Re_lp = 230
    #     f1 = 4.97 * (Re_lp ** (0.6049 - 1.064 / (theta ** 0.2))) * ((np.log((F_t / F_p) ** 0.5 + 0.9)) ** (-0.527))
    #     f2 = (((D_h / L_p) * np.log(0.3 * Re_lp)) ** (-2.966)) * ((F_p / L_l) ** (-0.7931 * T_p / T_h))
    #     f3 = ((T_p / D_m) ** (-0.0446)) * ((np.log(1.2 + (L_p / F_p) ** 1.4)) ** (-3.553)) * (theta ** (-0.477))
    #     f_230 = f1 * f2 * f3

    #     f_ar = (((1 + w) * f_130 ** 2 + (1 - w) * f_230 ** 2) / 2) ** 0.5

    Re_d_ar = rho_ar_in * V_ar_core * D_h / mu_ar_in / 10
    f_ar = 1.3525 * Re_d_ar ** (-0.449)

    # fator de colburn
    # j_ar = (Re_lp ** -0.49) * ((theta / 90) ** 0.27) * ((F_p / L_p) ** -0.14) * ((F_l / L_p) ** -0.29) * ((T_d / L_p) ** -0.23) * ((L_l / L_p) ** 0.68) * ((T_p / L_p) ** -0.28) * ((F_t / L_p) ** -0.05)
    j_ar = 0.1611 * (Re_d_ar ** -0.441)

    return j_ar, f_ar

def calor_tubo(G_f, fracao_vapor, fracao_bifasico, eta_aleta, cp_f_in, cp_f_out, \
                i_f_in, i_f_out, P_f_in, P_f_out, h_ar, A_total_elemento, m_ar, m_f, cp_ar_in, T_f_in, T_ar_in, \
                    x_f_in, x_f_out, fluido, L_elemento, P_i_tubo, rug, d_h_canal, BICUBIC_fluido, q_h):

    x_f_mean = (x_f_in + x_f_out) / 2
    P_f_mean = (P_f_in + P_f_out) / 2
    i_f_mean = (i_f_in + i_f_out) / 2
    cp_f_mean = (cp_f_in + cp_f_out) / 2
    
    Props_liquido = BICUBIC_fluido
    Props_vapor = BICUBIC_fluido
       
    if fracao_vapor == 1.0 and fracao_bifasico == 0.0:  # Somente vapor

        h_f = h_monofasico(G_f, fluido, d_h_canal, P_f_mean, i_f_mean, BICUBIC_fluido) # Coeficiente de transferência de calor
        UA = (1 / (h_f * L_elemento * P_i_tubo) + 1 / (eta_aleta * h_ar * A_total_elemento)) ** -1  # Calculo do UA, efetividade e calor total
        Q = calor_monofasico(m_ar, m_f, cp_ar_in, cp_f_mean, UA, T_ar_in, T_f_in)  # Calor
        DP = DP_monofasico(fluido, P_f_mean, i_f_mean, G_f, d_h_canal, L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        H_f = h_f
        
    elif fracao_vapor > 0.0 and fracao_bifasico > 0.0:  # se tem vapor superaquecido e escoamento bifásico
        
        # Propriedades do vapor para utilizar como média
        Props_vapor.update(CoolProp.PQ_INPUTS, P_f_in, 1.0)                    
        i_f_vapor = Props_vapor.hmass()
        cp_f_vapor = Props_vapor.cpmass()
        
        # Valores monofásicos
        h_f = h_monofasico(G_f, fluido, d_h_canal, P_f_mean, (i_f_in + i_f_vapor) / 2, BICUBIC_fluido)
        UA_mono = (1 / (h_f * L_elemento * P_i_tubo * fracao_vapor) + 1 / (eta_aleta * h_ar * A_total_elemento * fracao_vapor))**-1
        Q_mono = calor_monofasico(fracao_vapor * m_ar, m_f, cp_ar_in, (cp_f_in + cp_f_vapor) / 2, UA_mono, T_ar_in, T_f_in)
        DP_mono = DP_monofasico(fluido, P_f_mean, (i_f_in + i_f_vapor) / 2, G_f, d_h_canal, fracao_vapor * L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        H_f = h_f
        
        h_f = h_bifasico(G_f, fluido, d_h_canal, P_f_mean, x_f_mean, q_h, HEOS_fluido, BICUBIC_fluido)
        UA_bif = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico) + 1 / (eta_aleta * h_ar * A_total_elemento * fracao_bifasico))**-1
        Q_bifasico = calor_monofasico(fracao_bifasico * m_ar, m_f, cp_ar_in, cp_f_out, UA_bif, T_ar_in, T_f_in)
        DP_bif = DP_bifasico(fluido, P_f_in, i_f_in, (x_f_in + x_f_out) / 2, G_f, d_h_canal, fracao_bifasico * L_elemento, q_h, HEOS_fluido, BICUBIC_fluido)
        
        H_f += h_f
        UA = UA_mono + UA_bif
        Q = Q_mono + Q_bifasico
        DP = DP_mono + DP_bif

    elif fracao_vapor == 0.0 and fracao_bifasico == 1.0:  # só tem escoamento bifásico
        

        h_f = h_bifasico(G_f, fluido, d_h_canal, P_f_mean, x_f_mean, q_h, HEOS_fluido, BICUBIC_fluido)
        H_f = h_f      
        UA = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico) + 1 / (eta_aleta * h_ar * A_total_elemento * fracao_bifasico))**-1
        Q = calor_monofasico(fracao_bifasico * m_ar, m_f, cp_ar_in, cp_f_mean, UA, T_ar_in, T_f_in)
        DP = DP_bifasico(fluido, P_f_mean, i_f_mean, x_f_mean, G_f, d_h_canal, fracao_bifasico * L_elemento, q_h, HEOS_fluido, BICUBIC_fluido)
       
    elif fracao_vapor == 0 and fracao_bifasico > 0:  # Região com escoamento bifásico e líquido
        
        # Valores bifásico
        Props_liquido.update(CoolProp.PQ_INPUTS, P_f_in, 0)
        i_f_liq = Props_liquido.hmass()
        cp_f_liq = Props_liquido.cpmass()
        
        if x_f_mean > 0.02:
            h_f = h_bifasico(G_f, fluido, d_h_canal, P_f_mean, x_f_mean, q_h, HEOS_fluido, BICUBIC_fluido)
        else:
            h_f = h_monofasico(G_f, fluido, d_h_canal, P_f_mean, i_f_mean, BICUBIC_fluido)

        H_f = h_f
        UA_bif = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico) + 1 / (eta_aleta * h_ar * A_total_elemento * fracao_bifasico))**-1
        Q_bifasico = calor_monofasico(fracao_bifasico * m_ar, m_f, cp_ar_in, cp_f_in, UA_bif, T_ar_in, T_f_in)
        DP_bif = DP_bifasico(fluido, P_f_mean, (i_f_in + i_f_liq) / 2, (x_f_in + 0) / 2, G_f, d_h_canal, fracao_bifasico * L_elemento, HEOS_fluido, BICUBIC_fluido)
        
        # Valores monofásicos
        h_f = h_monofasico(G_f, fluido, d_h_canal, P_f_mean, (i_f_liq + i_f_out) / 2, BICUBIC_fluido)
        UA_mono = (1 / (h_f * L_elemento * P_i_tubo * (1 - fracao_bifasico)) + 1 / (eta_aleta * h_ar * A_total_elemento * (1 - fracao_bifasico)))**-1
        Q_mono = calor_monofasico((1 - fracao_bifasico) * m_ar, m_f, cp_ar_in, (cp_f_liq + cp_f_out) / 2, UA_mono, T_ar_in, T_f_in)
        DP_mono = DP_monofasico(fluido, P_f_mean, (i_f_liq + i_f_out) / 2, G_f, d_h_canal, (1 - fracao_bifasico) * L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        H_f += h_f
        
        UA = UA_mono + UA_bif
        Q = Q_mono + Q_bifasico
        DP = DP_mono + DP_bif

    elif fracao_vapor == 0 and fracao_bifasico == 0:  # só liquido
        
        h_f = h_monofasico(G_f, fluido, d_h_canal, P_f_mean, i_f_mean, BICUBIC_fluido)
        UA = (1 / (h_f * L_elemento * P_i_tubo) + 1 / (eta_aleta * h_ar * A_total_elemento))**-1
        Q = calor_monofasico(m_ar, m_f, cp_ar_in, cp_f_mean, UA, T_ar_in, T_f_in)
        DP = DP_monofasico(fluido, P_f_mean, i_f_mean, G_f, d_h_canal, L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        H_f = h_f

    return H_f, UA, DP, Q


def evaporador(Passes, m_f_total, N_elementos, N_canais, A_canal, P_f_inlet, T_f_inlet, x_f_inlet, fluido, P_ar_inlet, T_ar_inlet, V_ar_inlet, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo):
    
    # Propriedades dos fluidos
    global HEOS_fluido
    HEOS_fluido = CoolProp.AbstractState("HEOS", fluido)
    BICUBIC_fluido = CoolProp.AbstractState("BICUBIC", fluido)
    HEOS_Ar = CoolProp.AbstractState("HEOS", "Air")
    
    # Tensão superficial global, evitar tempo computacional
    HEOS_fluido.update(CoolProp.PQ_INPUTS, P_f_inlet, 1)
    global surface_tension_global
    surface_tension_global = HEOS_fluido.surface_tension()

    # Número de tubos na vertical
    N_v = sum(Passes)    
    
    # Propriedade do ar na entrada
    rho_ar_inlet = CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')
    
    # Área por tubo
    A_total_tubo = A_total / N_v
    A_min_tubo = A_min / N_v
    #A_aleta_tubo = A_aleta_total / N_v
    
    # Avaliando os elementos
    A_total_elemento = A_total_tubo / N_elementos
    A_min_elemento = A_min_tubo / N_elementos
    # A_aleta_elemento = A_aleta_tubo / N_elementos
    L_elemento = L_tubo / N_elementos   
    
    # Condutividade da aleta e rugosidade tubo
    #k_aleta = 237 
    rug = 0.000004e-3
    
    # Definição para o fluxo de massa e vazão mássica
    m_f = np.array([[m_f_total / Passes[i]]* N_elementos for i in range(len(Passes)) for _ in range(Passes[i])] )
    G_f = m_f / (N_canais * A_canal)

    # Temperatura de saturação
    T_f_sat = CP.PropsSI("T", "P", P_f_inlet, "Q", 0.5, fluido)
    
    # Condição inicial para o fluído, declaração de cada variável
    x_f_in, x_f_out = np.full((N_v, N_elementos), x_f_inlet), np.full((N_v, N_elementos), x_f_inlet)
    P_f_in, P_f_out = np.full((N_v, N_elementos), P_f_inlet), np.full((N_v, N_elementos), P_f_inlet)
    T_f_in, T_f_out = np.full((N_v, N_elementos), T_f_sat), np.full((N_v, N_elementos), T_f_sat)
    i_f_in, i_f_out = np.full((N_v, N_elementos), CP.PropsSI('H', 'P', P_f_inlet, 'Q', x_f_inlet, fluido)), np.full((N_v, N_elementos), CP.PropsSI('H', 'P', P_f_inlet, 'Q', x_f_inlet, fluido))
    cp_f_in, cp_f_out = np.full((N_v, N_elementos), CP.PropsSI('C', 'P', P_f_inlet, 'Q', x_f_inlet, fluido)), np.full((N_v, N_elementos), CP.PropsSI('C', 'P', P_f_inlet, 'Q', x_f_inlet, fluido))

    # Valores médios para o fluido
    x_f_mean = np.mean(np.array([x_f_in,x_f_out]), axis = 0)
    P_f_mean = np.mean(np.array([P_f_in,P_f_out]), axis = 0)
    i_f_mean = np.mean(np.array([i_f_in,i_f_out]), axis = 0)
    cp_f_mean = np.mean(np.array([cp_f_in,cp_f_out]), axis = 0)

    # Condição inicial para o ar
    P_ar_in, P_ar_out = np.full((N_v, N_elementos), P_ar_inlet), np.full((N_v, N_elementos), P_ar_inlet)
    T_ar_in, T_ar_out = np.full((N_v, N_elementos), T_ar_inlet), np.full((N_v,N_elementos), T_ar_inlet)
    T_ar_medio = (T_ar_in + T_ar_out) / 2
    cp_ar_in, cp_ar_out = np.full((N_v, N_elementos), CP.PropsSI('C', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')), np.full((N_v, N_elementos), CP.PropsSI('C', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air'))
    rho_ar_in, rho_ar_out = np.full((N_v, N_elementos), CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')), np.full((N_v, N_elementos), CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air'))
    mu_ar_in, mu_ar_out = np.full((N_v, N_elementos), CP.PropsSI('VISCOSITY', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')), np.full((N_v, N_elementos), CP.PropsSI('VISCOSITY', 'P', P_ar_inlet, 'T', T_ar_inlet,  'air'))
    Pr_ar_in, Pr_ar_out = np.full((N_v, N_elementos), CP.PropsSI('PRANDTL', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')), np.full((N_v, N_elementos), CP.PropsSI('PRANDTL', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air'))
    DP_ar = np.zeros((N_v, N_elementos))

    # Outras variáveis
    fracao_vapor, fracao_bifasico, DP, h_ar, UA, m_ar = np.zeros((N_v, N_elementos)), np.ones((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos))
    Q = np.ones((N_v, N_elementos))
    
    # H_f = np.zeros((N_v, N_elementos))
    
    # Valores antigos para estimativa dos erros
    i_f_in_old, i_f_out_old, T_ar_in_old, T_ar_out_old, P_f_in_old, P_f_out_old = np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v,N_elementos)), np.zeros((N_v,N_elementos)), np.zeros((N_v,N_elementos))
    cp_ar_old, Q_ar_old, Q_f_old = np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos))
    V_ar_core, G_ar = np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos))
    rho_ar_medio, f_ar, j_ar = np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos)), np.zeros((N_v, N_elementos))
    
    abc = 1
    Erro = [1000]
    start = time.time()
    tempo = [0]
    it = [0]
    
    Props_liquido = BICUBIC_fluido
    Props_vapor = BICUBIC_fluido
    
    # Vetorização de funções
    props_ar_vec = np.vectorize(propriedades_ar_vec)
    props_fluido_vec = np.vectorize(propriedades_fluido_vec)
    calor_tubo_vec = np.vectorize(calor_tubo)
    
    while (Erro[-1] > 1e-3):
               
        # Computo da velocidade do core
        V_ar_core = V_ar_inlet * (rho_ar_inlet / rho_ar_in) / sigma
        m_ar = V_ar_core * rho_ar_in * A_min_elemento
        G_ar = rho_ar_in * V_ar_core
        
        # Calculo do fator de atrito e colburn, h e DP
        j_ar, f_ar = fatores(V_ar_core, rho_ar_in, mu_ar_in, D_h)
        
        # Perda de carga lado do ar
        rho_ar_medio = ((1/2) * (1 / rho_ar_in + 1 / rho_ar_out))**-1
        DP_ar = (G_ar ** 2 / (2 * rho_ar_in)) * ((1 + sigma**2) * (rho_ar_in / rho_ar_out - 1) + f_ar * (A_total_elemento / (A_min_elemento)) * (rho_ar_in / rho_ar_medio))
        h_ar = G_ar * cp_ar_in * j_ar / (Pr_ar_in**(2/3))
        
        # eficiência da superficie aletada
        # TODO: Incluir eficiência da aleta caso necessário, incluir parâmetros na entrada da função
        # m_aleta = np.sqrt((2 * h_ar[i][j] / (k_aleta * e_aleta)) * (1 + e_aleta / F_l))
        # eta_aleta = np.tanh(m_aleta * l_aleta) / (m_aleta * l_aleta)
        # eta_aleta = 1 - (A_aleta_elemento / A_total_elemento) * (1 - eta_aleta)
        eta_aleta = 1
    
           
        # for i in range(N_v):
              
        #     for j in range(N_elementos):
 
        #         if fracao_vapor[i][j] == 1 and fracao_bifasico[i][j] == 0:  # Somente vapor
                     
        #             h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], i_f_mean[i][j], BICUBIC_fluido) # Coeficiente de transferência de calor
        #             UA[i][j] = (1 / (h_f * L_elemento * P_i_tubo) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento))**-1  # Calculo do UA, efetividade e calor total
        #             Q[i][j] = calor_monofasico(m_ar[i][j], m_f[i][j], cp_ar_in[i][j], cp_f_mean[i][j], UA[i][j], T_ar_in[i][j], T_f_in[i][j])  # Calor
        #             DP[i][j] = DP_monofasico(fluido, P_f_mean[i][j], i_f_mean[i][j], G_f[i][j], d_h_canal, L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        #             H_f[i,j] = h_f
                    
        #         elif fracao_vapor[i][j] > 0 and fracao_bifasico[i][j] > 0:  # se tem vapor superaquecido e escoamento bifásico
                    
        #             # Propriedades do vapor para utilizar como média
        #             Props_vapor.update(CoolProp.PQ_INPUTS, P_f_in[i][j], 1)                    
        #             i_f_vapor = Props_vapor.hmass()
        #             cp_f_vapor = Props_vapor.cpmass()
                    
        #             # Valores monofásicos
        #             h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], (i_f_in[i][j] + i_f_vapor) / 2, BICUBIC_fluido)
        #             UA_mono = (1 / (h_f * L_elemento * P_i_tubo * fracao_vapor[i][j]) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento * fracao_vapor[i][j]))**-1
        #             Q_mono = calor_monofasico(fracao_vapor[i][j] * m_ar[i][j], m_f[i][j], cp_ar_in[i][j], (cp_f_in[i][j] + cp_f_vapor) / 2, UA_mono, T_ar_in[i][j], T_f_in[i][j])
        #             DP_mono = DP_monofasico(fluido, P_f_mean[i][j], (i_f_in[i][j] + i_f_vapor) / 2, G_f[i][j], d_h_canal, fracao_vapor[i][j] * L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        #             H_f[i,j] = h_f
                    
        #             # Valores para o escoamento bifásico
        #             h_f = h_bifasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], (1 + x_f_out[i][j]) / 2, HEOS_fluido, BICUBIC_fluido)
        #             UA_bif = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico[i][j]) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento * fracao_bifasico[i][j]))**-1
        #             Q_bifasico = calor_monofasico(fracao_bifasico[i][j] * m_ar[i][j], m_f[i][j], cp_ar_in[i][j], cp_f_out[i][j], UA_bif, T_ar_in[i][j], T_f_in[i][j])
        #             DP_bif = DP_bifasico(fluido, P_f_in[i][j], i_f_in[i][j], (x_f_in[i][j] + x_f_out[i][j])/2, G_f[i][j], d_h_canal, fracao_bifasico[i][j] * L_elemento, HEOS_fluido, BICUBIC_fluido)
                    
        #             H_f[i,j] += h_f
        #             UA[i][j] = UA_mono + UA_bif
        #             Q[i][j] = Q_mono + Q_bifasico
        #             DP[i][j] = DP_mono + DP_bif
 
        #         elif fracao_vapor[i][j] == 0 and fracao_bifasico[i][j] == 1:  # só tem escoamento bifásico
                    
        #             if x_f_mean[i][j] > 0.02:
        #                 h_f = h_bifasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], x_f_mean[i][j], HEOS_fluido, BICUBIC_fluido)
        #             else:
        #                 h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], i_f_mean[i][j], BICUBIC_fluido)
                
        #             H_f[i,j] = h_f
        #             UA[i][j] = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico[i][j]) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento * fracao_bifasico[i][j]))**-1
        #             Q[i][j] = calor_monofasico(fracao_bifasico[i][j] * m_ar[i][j], m_f[i][j], cp_ar_in[i][j], cp_f_mean[i][j], UA[i][j], T_ar_in[i][j], T_f_in[i][j])
        #             DP[i][j] = DP_bifasico(fluido, P_f_mean[i][j], i_f_mean[i][j], x_f_mean[i][j], G_f[i][j], d_h_canal, fracao_bifasico[i][j] * L_elemento, HEOS_fluido, BICUBIC_fluido)
                   
        #         elif fracao_vapor[i][j] == 0 and fracao_bifasico[i][j] > 0:  # Região com escoamento bifásico e líquido
                    
        #             # Valores bifásico
        #             Props_liquido.update(CoolProp.PQ_INPUTS, P_f_in[i][j], 0)
        #             i_f_liq = Props_liquido.hmass()
        #             cp_f_liq = Props_liquido.cpmass()
                    
        #             if x_f_mean[i][j] > 0.02:
        #                 h_f = h_bifasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], x_f_mean[i][j], HEOS_fluido, BICUBIC_fluido)
        #             else:
        #                 h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], i_f_mean[i][j], BICUBIC_fluido)
    
        #             H_f[i,j] = h_f
        #             UA_bif = (1 / (h_f * L_elemento * P_i_tubo * fracao_bifasico[i][j]) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento * fracao_bifasico[i][j]))**-1
        #             Q_bifasico = calor_monofasico(fracao_bifasico[i][j] * m_ar[i][j], m_f[i][j], cp_ar_in[i][j], cp_f_in[i][j], UA_bif, T_ar_in[i][j], T_f_in[i][j])
        #             DP_bif = DP_bifasico(fluido, P_f_mean[i][j], (i_f_in[i][j] + i_f_liq) / 2, (x_f_in[i][j] + 0) / 2, G_f[i][j], d_h_canal, fracao_bifasico[i][j] * L_elemento, HEOS_fluido, BICUBIC_fluido)
                    
        #             # Valores monofásicos
        #             h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], (i_f_liq + i_f_out[i][j]) / 2, BICUBIC_fluido)
        #             UA_mono = (1 / (h_f * L_elemento * P_i_tubo * (1 - fracao_bifasico[i][j])) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento * (1 - fracao_bifasico[i][j])))**-1
        #             Q_mono = calor_monofasico((1 - fracao_bifasico[i][j]) * m_ar[i][j], m_f[i][j], cp_ar_in[i][j], (cp_f_liq + cp_f_out[i][j]) / 2, UA_mono, T_ar_in[i][j], T_f_in[i][j])
        #             DP_mono = DP_monofasico(fluido, P_f_mean[i][j], (i_f_liq + i_f_out[i][j]) / 2, G_f[i][j], d_h_canal, (1 - fracao_bifasico[i][j]) * L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        #             H_f[i,j] += h_f
                    
        #             UA[i][j] = UA_mono + UA_bif
        #             Q[i][j] = Q_mono + Q_bifasico
        #             DP[i][j] = DP_mono + DP_bif
 
        #         elif fracao_vapor[i][j] == 0 and fracao_bifasico[i][j] == 0:  # só liquido
                    
        #             h_f = h_monofasico(G_f[i][j], fluido, d_h_canal, P_f_mean[i][j], i_f_mean[i][j], BICUBIC_fluido)
        #             UA[i][j] = (1 / (h_f * L_elemento * P_i_tubo) + 1 / (eta_aleta * h_ar[i][j] * A_total_elemento))**-1
        #             Q[i][j] = calor_monofasico(m_ar[i][j], m_f[i][j], cp_ar_in[i][j], cp_f_mean[i][j], UA[i][j], T_ar_in[i][j], T_f_in[i][j])
        #             DP[i][j] = DP_monofasico(fluido, P_f_mean[i][j], i_f_mean[i][j], G_f[i][j], d_h_canal, L_elemento, rug, BICUBIC_fluido)  # Queda de pressão
        #             H_f[i,j] = h_f
        
        
        H_f, UA, DP, Q = calor_tubo_vec(G_f, fracao_vapor, fracao_bifasico, eta_aleta, cp_f_in, cp_f_out, \
                        i_f_in, i_f_out, P_f_in, P_f_out, h_ar, A_total_elemento, m_ar, m_f, cp_ar_in, T_f_in, T_ar_in, \
                            x_f_in, x_f_out, fluido, L_elemento, P_i_tubo, rug, d_h_canal, BICUBIC_fluido, Q / (L_elemento * P_i_tubo))
        
        Q_ar_old = m_ar * cp_ar_in * (T_ar_out - T_ar_in)
        Q_f_old = m_f * (i_f_in - i_f_out)
        
        i_f_out, i_f_in, P_f_in, P_f_out, T_ar_in, T_ar_out, T_ar_medio = serpentina(Passes, N_elementos, Q, i_f_out, i_f_in, P_f_out, P_f_in, DP, m_f, cp_ar_in, m_ar, T_ar_in, T_ar_out, DP_ar, P_ar_in, P_ar_out, N_v)             
        # x_f_in, x_f_out, cp_f_in, cp_f_out, T_f_in, T_f_out, fracao_vapor, fracao_bifasico = propriedades_fluido(P_f_in, P_f_out, i_f_in, i_f_out, Q, m_f, N_v, N_elementos, BICUBIC_fluido)        
        # rho_ar_in, rho_ar_out, cp_ar_in, cp_ar_out, mu_ar_in, mu_ar_out, Pr_ar_in, Pr_ar_out = propriedades_ar(P_ar_in, P_ar_out, T_ar_in, T_ar_out, N_v, N_elementos, HEOS_Ar)      
        rho_ar_in, rho_ar_out, cp_ar_in, cp_ar_out, mu_ar_in, mu_ar_out, Pr_ar_in, Pr_ar_out = props_ar_vec(P_ar_in, P_ar_out, T_ar_in, T_ar_out, HEOS_Ar)
        x_f_in, x_f_out, cp_f_in, cp_f_out, T_f_in, T_f_out, fracao_vapor, fracao_bifasico = props_fluido_vec(P_f_in, P_f_out, i_f_in, i_f_out, Q, m_f, BICUBIC_fluido)
 
        Q_ar = m_ar * cp_ar_in * (T_ar_out - T_ar_in)
        Q_f = m_f * (i_f_in - i_f_out)
        
        x_f_mean = np.mean(np.array([x_f_in,x_f_out]), axis = 0)
        P_f_mean = np.mean(np.array([P_f_in,P_f_out]), axis = 0)
        i_f_mean = np.mean(np.array([i_f_in,i_f_out]), axis = 0)
        cp_f_mean = np.mean(np.array([cp_f_in,cp_f_out]), axis = 0)
        
        # Erros
        Residuo_ar = Q_ar - Q_ar_old
        Residuo_fluido = Q_f - Q_f_old
        Residuo_ar_fluido = Q_ar - Q_f
        
        Erro_fluido = np.sqrt(sum(sum(Residuo_fluido ** 2)))
        Erro_ar = np.sqrt(sum(sum(Residuo_ar ** 2)))
        Erro_ar_fluido = np.sqrt(sum(sum(Residuo_ar_fluido ** 2)))
        
        Erro.append(max(Erro_fluido, Erro_ar, Erro_ar_fluido))
        tempo.append(time.time()-start)
        it.append(it[-1]+1)
                
        if abc > 50:
            print("Atenção: Valor não convergiu!")
            break
        
        abc += 1
    
    Q_total = sum(sum(Q))
    print("Q_total: {}".format(Q_total))
 
    Output = namedtuple('Output', 'Q_total T_f_in f_vapor f_bifasico x_in x_out UA i_f_in i_f_out Q P_f_in P_f_out m_f G_f h_f h_ar')
    HX = Output(
        Q_total = Q_total,
        T_f_in = T_f_in,
        f_vapor = fracao_vapor,
        f_bifasico = fracao_bifasico,
        x_in = x_f_in,
        x_out = x_f_out,
        UA = UA,    
        i_f_in = i_f_in,
        i_f_out = i_f_out,
        Q = Q,
        P_f_in = P_f_in,
        P_f_out = P_f_out,
        m_f = m_f,
        G_f = G_f,
        h_f = H_f,
        h_ar = h_ar
        )
    
    return HX


def main():
    # # Geometria do trocador
    N_v = 16
    N_c = 1
    L_tubo = 228e-3  # Comprimento do tubo
    H = 178e-3  # altura do trocador
    Comprimento = 18e-3
    A_f = L_tubo * H
    
    # # Geometria do tubo/canal
    N_canais = 20
    D_m = 2e-3
    T_d = Comprimento
    T_p = 10e-3
    T_h = T_p - D_m  
    H_canal = 0.7e-3
    L_canal = 0.7e-3
    A_canal = H_canal * L_canal
    P_canal = 2 * (L_canal + H_canal)
    d_h_canal = 4 * A_canal / P_canal
    P_i_tubo = 2 * (N_canais * L_canal + H_canal)
    
    # # Geometria das aletas e louver
    e_aleta = 0.100e-3
    N_aleta = 140
    F_p = L_tubo / N_aleta
    N_p = N_v + 1
    F_l = 8e-3
    L_l = 7e-3
    L_p = 1e-3
    F_d = Comprimento
    F_t = e_aleta
    theta = 20
    N_f = N_p * N_aleta
    l_aleta = 0.5 * (T_h ** 2 + F_t ** 2) ** 0.5 - F_t  # fin efficiency
    
    # # Avaliando uma célula
    A_p_cell = 2 * T_d * (F_p - e_aleta) + 2 * F_p * D_m
    A_f_cell = 2 * F_d * ((T_h ** 2 + F_p ** 2) ** 0.5 - e_aleta)
    A_cell = A_p_cell + A_f_cell
    A_o_cell = F_p * T_h - e_aleta * ((T_h ** 2 + F_p ** 2) ** 0.5 - e_aleta)
    A_fr_cell = F_p * (T_h + D_m)
    V_cell = F_p * T_p * F_d
    # sigma = A_o_cell / A_fr_cell
    beta = A_cell / V_cell
    D_h = 4 * A_o_cell * F_d / A_cell
    A_aleta_total = A_f_cell * N_aleta * (N_v + 1)
    
    # # Avaliando o trocador todo
    Vol = Comprimento * H * L_tubo
    # A_min = sigma * A_f
    A_min = A_f - ((N_v) * D_m * L_tubo + (N_v) * N_aleta * e_aleta * T_h)
    sigma = A_min / A_f
    A_total = beta * Vol
    
    # Dados_relatorio = {
    #         "T_ar": [19.3, 20.6, 20.4, 20.4, 20.3, 20.1, 18.8, 18.9, 19.0],
    #         "Vazao_ar": [0.0144, 0.0216, 0.0269, 0.0335, 0.0395, 0.045, 0.0567, 0.0843, 0.1120],
    #         "T_agua": [40.4, 40.0, 39.8, 39.5, 39.5, 40.0, 41.2, 40.5, 40.4],
    #         "Vazao_agua": [73.8, 98.4, 124.4, 145.0, 170.9, 201.5, 250.5, 296.5, 299.1],
    #         }
    
    # Teste = 9
    # m_ar_inlet = Dados_relatorio["Vazao_ar"][Teste-1]
    # T_ar_inlet = Dados_relatorio["T_ar"][Teste-1] + 273.15
    # m_f_total = Dados_relatorio["Vazao_agua"][Teste-1] / 3600
    # T_f_sat = Dados_relatorio["T_agua"][Teste-1] + 273.15
    
    # # # Propriedades do ar na entrada
    # P_ar_inlet = 101.3e3
    # rho_ar_inlet = CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')
    # V_ar_inlet = m_ar_inlet / (rho_ar_inlet * A_f)
    
    # # Propriedades do fluido na entrada
    # fluido = 'R134a'
    # T_f_sat = 40.4 + 273.15
    # P_f_inlet = CP.PropsSI('P', 'T', T_f_sat, 'Q', 0.5, fluido)
    # T_f_inlet = T_f_sat + 10
    # m_f_total = (23 / 3600) #total na entrada
    
    # # Propriedades do fluido na entrada
    # # fluido = 'water'
    # # P_f_inlet = 101.3e3
    # # T_f_inlet = T_f_sat
    
        
    # #arranjo de passes
    # Passes = np.array([8, 5, 3])  # Incluido no outro programa
    # N_elementos = 12
    
    # Novo trocador de calor
    Passes = np.array([25, 15, 10])  # Incluido no outro programa
    N_elementos = 16
    N_v = sum(Passes)
    L_tubo = 500e-3
    H = 11.125 * N_v * 1e-3
    A_f = L_tubo * H
    sigma = 0.7760693869505224
    beta = 1200.2674552928931
    A_min = sigma * A_f
    A_total = beta * A_f * Comprimento
    D_h = 4 * sigma / beta

    # Propriedades do fluido na entrada
    fluido = 'R134a'
    T_f_sat = 5 + 273.15
    x_f_inlet = 0.4
    P_f_inlet = CP.PropsSI('P', 'T', T_f_sat, 'Q', 0.5, fluido)
    T_f_inlet = T_f_sat
    m_f_total = (80 / 3600) #total na entrada

    # Propriedades do ar na entrada
    T_ar_inlet = 10 + 273.15
    P_ar_inlet = 101.3e3
    V_ar_inlet = 10/3.6
    
    Evaporador = evaporador(Passes, m_f_total, N_elementos, N_canais, A_canal, P_f_inlet, T_f_inlet, x_f_inlet, fluido, P_ar_inlet, T_ar_inlet, V_ar_inlet, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo)
    
    return Evaporador


if __name__ == "__main__":
    
    # lprofiler = LineProfiler()
    # lprofiler.add_function(evaporador)
    # # lprofiler.add_function(DP_bifasico)
    # lprofiler.add_function(propriedades_fluido)
    # lp_wrapper = lprofiler(main)
    
    # lp_wrapper()
    # lprofiler.print_stats()
   
    start = time.time()
    Evaporador = main()
    finish = time.time()
    print(finish - start)

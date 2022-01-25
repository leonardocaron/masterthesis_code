# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:32:18 2021

@author: Usuário
"""

import Condensador
import Compressor
import numpy as np
import CoolProp.CoolProp as CP
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def erro_condensador(P_cond, P_evap):
    T_evap = CP.PropsSI('T', 'P', P_evap, 'Q', 0.5, fluido)
    T_cond = CP.PropsSI('T', 'P', P_cond, 'Q', 0.5, fluido)
    T_suc = T_evap + 1
    h_1 = CP.PropsSI('H', 'P', P_evap, 'T', T_suc, fluido)                 
    m_dot, W_dot = Compressor.compressor_sierra_pol(5000, T_evap-273.15, T_cond-273.15)
    h_2 = W_dot / (m_dot / 3600) + h_1
    T_inlet = CP.PropsSI('T', 'P', P_cond, 'H', h_2, fluido)
    start = time.time()
    HX = Condensador.condensador(Passes, m_dot / 3600, N_elementos, N_canais, A_canal, P_cond, T_inlet, fluido, P_ar_inlet, T_ar_inlet, V_ar_inlet, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo)
    finish = time.time()
    print(finish - start)
    Q_cond_hx = HX.Q_total
    
    T_cond_out = T_cond - 1
    h_3 = CP.PropsSI('H', 'P', P_cond, 'T', T_cond_out, fluido)
    Q_cond_cycle = (m_dot / 3600) * (h_2 - h_3)
    Erro_cond = Q_cond_hx - Q_cond_cycle
    # print(Erro_cond)
    
    return Erro_cond

# Geometria do tubo/canal
N_canais = 20
H_canal = 0.7e-3
L_canal = 0.7e-3
A_canal = H_canal * L_canal
P_canal = 2 * (L_canal + H_canal)
d_h_canal = 4 * A_canal / P_canal
P_i_tubo = 2 * (N_canais * L_canal + H_canal)

# Novo trocador de calor
Passes = np.array([25, 15, 10])  # Incluido no outro programa
N_v = sum(Passes)
L_tubo = 500e-3
H = 11.125 * N_v * 1e-3
Comprimento = 18e-3
A_f = L_tubo * H
sigma = 0.7760693869505224
beta = 1200.2674552928931
A_min = sigma * A_f
A_total = beta * A_f * Comprimento
D_h = 4 * sigma / beta

# Propriedades do fluido na entrada
fluido = 'R134a'
T_f_sat = 45.4 + 273.15
P_f_inlet = CP.PropsSI('P', 'T', T_f_sat, 'Q', 0.5, fluido)
T_f_inlet = T_f_sat + 30
m_f_total = (80 / 3600) #total na entrada

# Propriedades do ar na entrada
T_ar_inlet = 35 + 273.15
P_ar_inlet = 101.3e3
rho_ar_inlet = CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')
m_ar_inlet = 0.1120
V_ar_inlet = m_ar_inlet / (rho_ar_inlet * A_f)
V_ar_inlet = 10/3.6

N_elementos = 20


# Usando newton-raphson
T_cond = 45 + 273.15
P_cond = CP.PropsSI('P', 'T', T_cond, 'Q', 0.5, fluido)
T_evap = 0 + 273.15
P_evap = CP.PropsSI('P', 'T', T_evap, 'Q', 0.5, fluido)

deltaP = 10
f_x = erro_condensador(P_cond, P_evap)
f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
f_linha_x = (f_x_dX - f_x) / deltaP 
P_cond_new = P_cond - f_x / f_linha_x
T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

P_cond = P_cond_new
f_x = erro_condensador(P_cond, P_evap)
f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
f_linha_x = (f_x_dX - f_x) / deltaP 
P_cond_new = P_cond - f_x / f_linha_x
T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)




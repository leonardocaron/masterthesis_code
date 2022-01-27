# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:32:18 2021

@author: Usuário
"""

import Condensador
import Compressor
import Evaporador
import numpy as np
import CoolProp.CoolProp as CP
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def erro_condensador(P_cond, P_evap, delta_sh, delta_sc, m_dot, W_dot, h_evap_out):
      
    # Cálculo da temperatura de descarga
    T_evap = CP.PropsSI('T', 'P', P_evap, 'Q', 0.5, fluido) 
    T_cond = CP.PropsSI('T', 'P', P_cond, 'Q', 0.5, fluido) 

    h_2 = W_dot / m_dot + h_evap_out
    T_f_inlet = CP.PropsSI('T', 'P', P_cond, 'H', h_2, fluido)

    # Simulação condensador
    HX = Condensador.condensador(Passes, m_dot, N_elementos, N_canais, A_canal, P_cond, T_f_inlet, fluido, P_ar_inlet, T_ar_inlet_cond, V_ar_inlet_cond, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo)
   
    # Cálculo do erro
    T_cond_out = T_cond - delta_sc
    h_3 = CP.PropsSI('H', 'P', P_cond, 'T', T_cond_out, fluido)
    Q_cond_cycle = m_dot * (h_2 - h_3)
    
    Erro_cond = HX.Q_total - Q_cond_cycle
    
    # Dados de saída do condensador
    T_cond_out = np.mean(HX.T_f_out[-Passes[-1]:,-1])
    h_cond_out = np.mean(HX.i_f_out[-Passes[-1]:,-1])
    
    return Erro_cond, T_cond_out, h_cond_out, HX.Q_total

def erro_evaporador(P_cond, P_evap, delta_sh, delta_sc, m_dot, W_dot, x_inlet):
      
    # Cálculo da temperatura de descarga
    T_evap = CP.PropsSI('T', 'P', P_evap, 'Q', 0.5, fluido) 
    T_cond = CP.PropsSI('T', 'P', P_cond, 'Q', 0.5, fluido) 

    # Simulação condensador
    HX = Evaporador.evaporador(Passes, m_dot, N_elementos, N_canais, A_canal, P_evap, T_evap, x_inlet, fluido, P_ar_inlet, T_ar_inlet_evap, V_ar_inlet_evap, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo)
 
    # Cálculo do erro
    T_evap_out = T_evap + delta_sh
    h_3 = CP.PropsSI('H', 'P', P_evap, 'Q', x_inlet, fluido)
    h_4 = CP.PropsSI('H', 'P', P_evap, 'T', T_evap_out, fluido)
    Q_evap_cycle = m_dot * (h_4 - h_3)
    
    Erro_evap = HX.Q_total - Q_evap_cycle
    # print(Erro_evap)
    
    # Dados de saída do condensador
    T_evap_out = np.mean(HX.T_f_out[-Passes[-1]:,-1])
    h_evap_out = np.mean(HX.i_f_out[-Passes[-1]:,-1])
    
    return Erro_evap, T_evap_out, h_evap_out, HX.Q_total


# Geometria do tubo/canal
N_canais = 20
H_canal = 0.7e-3
L_canal = 0.7e-3
A_canal = H_canal * L_canal
P_canal = 2 * (L_canal + H_canal)
d_h_canal = 4 * A_canal / P_canal
P_i_tubo = 2 * (N_canais * L_canal + H_canal)

# Novo trocador de calor
Passes = np.array([20, 20, 20])  # Incluido no outro programa
N_v = sum(Passes)
L_tubo = 400e-3
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

# Propriedades do ar na entrada do condensador
T_ar_inlet_cond = 30 + 273.15
P_ar_inlet = 101.3e3
V_ar_inlet_cond = 10/3.6
N_elementos = 15

# Propriedades do ar na entrada do evaporador
T_ar_inlet_evap = 15 + 273.15
P_ar_inlet = 101.3e3
V_ar_inlet_evap = 5/3.6
N_elementos = 15

delta_sh = 3
delta_sc = 2
    
# Loop do condensador

# Usando método da secante
# Estimativa inicial, 2 pontos
T_cond = np.array([40 + 273.15, 40.5 + 273.15])
T_evap = np.array([3 + 273.15, 3.5 + 273.15])
P_cond = np.array([CP.PropsSI('P', 'T', T_cond[0], 'Q', 0.5, fluido), CP.PropsSI('P', 'T', T_cond[1], 'Q', 0.5, fluido)])
P_evap = np.array([CP.PropsSI('P', 'T', T_evap[0], 'Q', 0.5, fluido), CP.PropsSI('P', 'T', T_evap[1], 'Q', 0.5, fluido)])
h_evap_out = CP.PropsSI('H', 'P', P_evap[-1], 'T', T_evap[-1] + delta_sh, fluido)
m_dot = np.array([Compressor.compressor_sierra_pol(5000, T_evap[0] - 273.15, T_cond[0] - 273.15)[0], Compressor.compressor_sierra_pol(5000, T_evap[1] - 273.15, T_cond[1] - 273.15)[0]]) / 3600
W_dot = np.array([Compressor.compressor_sierra_pol(5000, T_evap[0] - 273.15, T_cond[0] - 273.15)[1], Compressor.compressor_sierra_pol(5000, T_evap[1] - 273.15, T_cond[1] - 273.15)[1]])
fx_cond = np.array([erro_condensador(P_cond[0], P_evap[0], delta_sh, delta_sc, m_dot[0], W_dot[0], h_evap_out)[0], \
               erro_condensador(P_cond[1], P_evap[1], delta_sh, delta_sc, m_dot[1], W_dot[1], h_evap_out)[0]])
    
fx_evap = np.array([erro_evaporador(P_cond[0], P_evap[0], delta_sh, delta_sc, m_dot[0], W_dot[0], 0.3)[0], \
                    erro_evaporador(P_cond[1], P_evap[1], delta_sh, delta_sc, m_dot[1], W_dot[1], 0.3)[0]])



start = time.time()
index_cond = 2
index_evap = 2
# index = 2
Erro = 1
for it in range(50):
            
    # Loop do condensador
    while(abs(fx_cond[-1]) > 1e-3):
        
        # Derivada na função
        fx_linha_cond = (fx_cond[index_cond - 1] - fx_cond[index_cond - 2]) / (P_cond[index_cond - 1] - P_cond[index_cond - 2])

        P_cond = np.append(P_cond, P_cond[index_cond - 1] - fx_cond[index_cond - 1] / fx_linha_cond)
        T_cond = np.append(T_cond, CP.PropsSI('T', 'P', P_cond[-1], 'Q', 0.5, fluido))
        
        m_dot = np.append(m_dot, Compressor.compressor_sierra_pol(5000, T_evap[-1] - 273.15, T_cond[-1] - 273.15)[0] / 3600)
        W_dot = np.append(W_dot, Compressor.compressor_sierra_pol(5000, T_evap[-1] - 273.15, T_cond[-1] - 273.15)[1])
        
        Dados_cond = erro_condensador(P_cond[-1], P_evap[-1], delta_sh, delta_sc, m_dot[-1], W_dot[-1], h_evap_out)
        fx_cond = np.append(fx_cond, Dados_cond[0])
      
        index_cond +=1 
       
        
    # Loop do evaporador
    while(abs(fx_evap[-1]) > 1e-3):
        
        # Derivada na função
        fx_linha_evap = (fx_evap[index_evap - 1] - fx_evap[index_evap - 2]) / (P_evap[index_evap - 1] - P_evap[index_evap - 2])
    
        # Valor obtido
        P_evap = np.append(P_evap, P_evap[index_evap - 1] - fx_evap[index_evap - 1] / fx_linha_evap)
        T_evap = np.append(T_evap, CP.PropsSI('T', 'P', P_evap[-1], 'Q', 0.5, fluido))
    
        # Título na entrada do evaporador
        x_f_inlet = CP.PropsSI('Q', 'P', P_evap[-1], 'H', Dados_cond[2], fluido)

        Dados_evap = erro_evaporador(P_cond[-1], P_evap[-1], delta_sh, delta_sc, m_dot[-1], W_dot[-1], x_f_inlet)
        fx_evap = np.append(fx_evap, Dados_evap[0])
        h_evap_out = Dados_evap[2]
        
        index_evap += 1

    
    Erro = Dados_evap[3] + W_dot[-1] - Dados_cond[3]
    if abs(Erro) < 0.5:
        break
    
    m_dot = np.append(m_dot, Compressor.compressor_sierra_pol(5000, T_evap[-1] - 273.15, T_cond[-1] - 273.15)[0] / 3600)
    W_dot = np.append(W_dot, Compressor.compressor_sierra_pol(5000, T_evap[-1] - 273.15, T_cond[-1] - 273.15)[1])
    Dados_cond = erro_condensador(P_cond[-1], P_evap[-1], delta_sh, delta_sc, m_dot[-1], W_dot[-1], h_evap_out)    
    fx_cond[-1] = Dados_cond[0]
    Dados_evap = erro_evaporador(P_cond[-1], P_evap[-1], delta_sh, delta_sc, m_dot[-1], W_dot[-1], x_f_inlet)
    fx_evap[-1] = Dados_evap[0]

finish = time.time()
print(finish - start)


# f_linha = (f_x_2 - f_x_1) / (P_cond_2 - P_cond_1)
# P_cond_3 = P_cond_2 - f_x_2 / f_linha
# T_cond_3 = CP.PropsSI('T', 'P', P_cond_3, 'Q', 0.5, fluido)
# P_evap_3 = P_evap_2
# f_x_3 = erro_condensador(P_cond_3, P_evap_3)

# f_linha = (f_x_3 - f_x_2) / (P_cond_3 - P_cond_2)
# P_cond_4 = P_cond_3 - f_x_3 / f_linha
# T_cond_4 = CP.PropsSI('T', 'P', P_cond_4, 'Q', 0.5, fluido)
# P_evap_4 = P_evap_3
# f_x_4 = erro_condensador(P_cond_4, P_evap_4)

# f_linha = (f_x_4 - f_x_3) / (P_cond_4 - P_cond_3)
# P_cond_5 = P_cond_4 - f_x_4 / f_linha
# T_cond_5 = CP.PropsSI('T', 'P', P_cond_5, 'Q', 0.5, fluido)
# P_evap_5 = P_evap_4
# f_x_5 = erro_condensador(P_cond_5, P_evap_5)

# f_linha = (f_x_5 - f_x_4) / (P_cond_5 - P_cond_4)
# P_cond_6 = P_cond_5 - f_x_5 / f_linha
# T_cond_6 = CP.PropsSI('T', 'P', P_cond_6, 'Q', 0.5, fluido)
# P_evap_6 = P_evap_5
# f_x_6 = erro_condensador(P_cond_6, P_evap_6)

# deltaP = 10
# f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# f_linha_x = (f_x_dX - f_x) / deltaP 
# P_cond_new = P_cond - f_x / f_linha_x
# T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

# P_cond = P_cond_new
# f_x = erro_condensador(P_cond, P_evap)
# f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# f_linha_x = (f_x_dX - f_x) / deltaP 
# P_cond_new = P_cond - f_x / f_linha_x
# T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

# # P_cond = P_cond_new
# # f_x = erro_condensador(P_cond, P_evap)
# # f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# # f_linha_x = (f_x_dX - f_x) / deltaP 
# # P_cond_new = P_cond - f_x / f_linha_x
# # T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

# # P_cond = P_cond_new
# # f_x = erro_condensador(P_cond, P_evap)
# # f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# # f_linha_x = (f_x_dX - f_x) / deltaP 
# # P_cond_new = P_cond - f_x / f_linha_x
# # T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

# # P_cond = P_cond_new
# # f_x = erro_condensador(P_cond, P_evap)
# # f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# # f_linha_x = (f_x_dX - f_x) / deltaP 
# # P_cond_new = P_cond - f_x / f_linha_x
# # T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)

# # P_cond = P_cond_new
# # f_x = erro_condensador(P_cond, P_evap)
# # f_x_dX = erro_condensador(P_cond + deltaP, P_evap)
# # f_linha_x = (f_x_dX - f_x) / deltaP 
# # P_cond_new = P_cond - f_x / f_linha_x
# # T_cond = CP.PropsSI('T', 'P', P_cond_new, 'Q', 0.5, fluido)
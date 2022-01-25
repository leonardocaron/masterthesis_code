# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:32:18 2021

@author: Usuário
"""

import Condensador
import numpy as np
import CoolProp.CoolProp as CP
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Geometria do trocador
Passes = np.array([8, 5, 3])  # Incluido no outro programa
N_v = sum(Passes)
# N_c = 1
# N_total = N_v * N_c
L_tubo = 228e-3  # Comprimento do tubo
H = 178e-3  # altura do trocador
Comprimento = 18e-3
A_f = L_tubo * H

# Geometria do tubo/canal
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

# Geometria das aletas e louver
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

# Avaliando uma célula
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

# Avaliando o trocador todo
Vol = Comprimento * H * L_tubo
# A_min = sigma * A_f
A_min = A_f - ((N_v) * D_m * L_tubo + (N_v) * N_aleta * e_aleta * T_h)
sigma = A_min / A_f
A_total = beta * Vol

Dados_relatorio = {
        "T_ar": [19.3, 20.6, 20.4, 20.4, 20.3, 20.1, 18.8, 18.9, 19.0],
        "Vazao_ar": [0.0144, 0.0216, 0.0269, 0.0335, 0.0395, 0.045, 0.0567, 0.0843, 0.1120],
        "T_agua": [40.4, 40.0, 39.8, 39.5, 39.5, 40.0, 41.2, 40.5, 40.4],
        "Vazao_agua": [73.8, 98.4, 124.4, 145.0, 170.9, 201.5, 250.5, 296.5, 299.1],
        }

Teste = 9
m_ar_inlet = Dados_relatorio["Vazao_ar"][Teste-1]
T_ar_inlet = Dados_relatorio["T_ar"][Teste-1] + 273.15
m_f_total = Dados_relatorio["Vazao_agua"][Teste-1] / 3600
T_f_sat = Dados_relatorio["T_agua"][Teste-1] + 273.15

# Propriedades do ar na entrada
P_ar_inlet = 101.3e3
rho_ar_inlet = CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')
V_ar_inlet = m_ar_inlet / (rho_ar_inlet * A_f)
#V_ar_inlet = 50/3.6

# Propriedades do fluido na entrada
fluido = 'R134a'
T_f_sat = 40.4 + 273.15
P_f_inlet = CP.PropsSI('P', 'T', T_f_sat, 'Q', 0.5, fluido)
T_f_inlet = T_f_sat + 15
m_f_total = (23 / 3600) #total na entrada

# Propriedades do fluido na entrada
# fluido = 'water'
# P_f_inlet = 101.3e3
# T_f_inlet = T_f_sat

tempo = []
dif = []
Q = []

# Novo trocador de calor
Passes = np.array([25, 15, 10])  # Incluido no outro programa
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
T_f_sat = 45.4 + 273.15
P_f_inlet = CP.PropsSI('P', 'T', T_f_sat, 'Q', 0.5, fluido)
T_f_inlet = T_f_sat + 30
m_f_total = (80 / 3600) #total na entrada

# Propriedades do ar na entrada
T_ar_inlet = 35 + 273.15
P_ar_inlet = 101.3e3
rho_ar_inlet = CP.PropsSI('D', 'P', P_ar_inlet, 'T', T_ar_inlet, 'air')
V_ar_inlet = m_ar_inlet / (rho_ar_inlet * A_f)
V_ar_inlet = 10/3.6


N_elementos = list(range(2,50,2))
for i, N_elemento in enumerate(N_elementos):
    
    start = time.time()
    HX = Condensador.condensador(Passes, m_f_total, N_elemento, N_canais, A_canal, P_f_inlet, T_f_inlet, fluido, P_ar_inlet, T_ar_inlet, V_ar_inlet, A_f, sigma, D_h, d_h_canal, P_i_tubo, A_total, A_min, L_tubo)
    finish = time.time()
    
    fracao_vapor = HX.f_vapor
    fracao_bifasico = HX.f_bifasico
    x_f_in = HX.x_in
    x_f_out = HX.x_out
    UA = HX.UA
    T_f_in = HX.T_f_in
    prop = HX.i_f_out
    h_ar = HX.h_ar
    h_f = HX.h_f


    tempo.append(finish-start)
    Q.append(HX.Q_total)
    if not i == 0:
        dif.append(100*abs(Q[i]-Q[i-1])/Q[i])

    fracao_vapor_passes = np.empty((len(Passes),fracao_vapor.shape[1]))
    fracao_bifasico_passes = np.empty((len(Passes),fracao_vapor.shape[1]))
    x_in = np.empty((len(Passes),fracao_vapor.shape[1]))
    x_out = np.empty((len(Passes),fracao_vapor.shape[1]))
    HTC = np.empty((len(Passes),fracao_vapor.shape[1]))
    Prop = np.empty((len(Passes),fracao_vapor.shape[1]))
    H_ar = np.empty((len(Passes),fracao_vapor.shape[1]))
    H_f = np.empty((len(Passes),fracao_vapor.shape[1]))
  
    posição = 0
    for passe, tubos in enumerate(Passes):     
        fracao_vapor_passes[passe,:] = np.mean(fracao_vapor[posição:posição+tubos, :], axis = 0)
        fracao_bifasico_passes[passe,:] = np.mean(fracao_bifasico[posição:posição+tubos, :], axis = 0)
        x_in[passe,:] = np.mean(x_f_in[posição:posição+tubos, :], axis = 0)
        x_out[passe,:] = np.mean(x_f_out[posição:posição+tubos, :], axis = 0)
        HTC[passe,:] = np.mean(UA[posição:posição+tubos, :], axis = 0)
        Prop[passe,:] = np.mean(prop[posição:posição+tubos, :], axis = 0)
        H_ar[passe,:] = np.mean(h_ar[posição:posição+tubos, :], axis = 0)
        H_f[passe,:] = np.mean(h_f[posição:posição+tubos, :], axis = 0)
        posição += tubos
       
    fracao_vapor_passes = fracao_vapor_passes.reshape(1,len(Passes) * fracao_vapor.shape[1])[0]
    fracao_bifasico_passes = fracao_bifasico_passes.reshape(1,len(Passes) * fracao_vapor.shape[1])[0]
    x_in = x_in.reshape(1,len(Passes) * fracao_vapor.shape[1])[0]
    x_out = x_out.reshape(1,len(Passes) * fracao_vapor.shape[1])[0]
    HTC = HTC.reshape(1,len(Passes) * HTC.shape[1])[0]
    Prop = Prop.reshape(1,len(Passes) * Prop.shape[1])[0]
    H_ar = H_ar.reshape(1,len(Passes) * H_ar.shape[1])[0]
    H_f = H_f.reshape(1,len(Passes) * H_f.shape[1])[0]
    
    título = []
    comprimento = []
    comp_HTC = []
    L_elemento = L_tubo / N_elemento
    
    for i in range(len(x_in)):
        # Só vapor
        if fracao_vapor_passes[i] == 1 and fracao_bifasico_passes[i] == 0:
            título.append(x_in[i])
            comprimento.append(i*L_elemento)             
        #Verifica se tem vapor e bifasico    
        elif fracao_vapor_passes[i] > 0 and fracao_bifasico_passes[i] > 0:
            #Inclui título de vapor
            título.append(x_in[i])          
            comprimento.append(i*L_elemento)                     
            #Faz computo para inicio do bifasico
            título.append(x_in[i])
            comprimento.append(i*L_elemento + L_elemento*fracao_vapor_passes[i])   
        elif fracao_vapor_passes[i] == 0 and fracao_bifasico_passes[i] == 1:          
            título.append(x_in[i])          
            comprimento.append(i*L_elemento)                        
        # Computo do valor bifasico
        elif fracao_vapor_passes[i] == 0 and fracao_bifasico_passes[i] > 0:
            #Inclui título de saída
            título.append(x_in[i])
            comprimento.append(i*L_elemento)
            #Faz computo para inicio do bifasico
            título.append(x_out[i])           
            comprimento.append(i*L_elemento + L_elemento*fracao_bifasico_passes[i])           
        # Só líquido
        else:
            título.append(x_in[i])           
            comprimento.append(i*L_elemento)          
        comp_HTC.append(L_elemento/2 + i*L_elemento)
                  
    título.append(x_out[-1])
    comprimento.append(len(x_in)*L_elemento)
    
    #Distribuição do título
    plt.figure(dpi=200)
    plt.plot(comprimento, título, color='black')
    linhas = [i * L_tubo for i in range(len(Passes)+1)]
    for linha in linhas:   
        plt.axvline(x = linha, color = 'black', linestyle='-.', linewidth = 0.5)    
    plt.xlim([0,L_tubo*len(Passes)])
    plt.title(f"$N_{{elementos}}$ = {N_elemento}")
    plt.ylabel("Título [-]")
    plt.xlabel("Comprimento do trocador [m]")
    
    # # Imprime linha onde título == 1
    # título = np.array(título)
    # pos = comprimento[np.where(título == 1)[0][-1]]
    # valor = título[np.where(título == 1)[0][-1]]
    # plt.scatter(pos,valor, s = 8, color='red')
    # plt.text(pos + 0.01, 1, f'L = {pos*1e3:.2f}mm')
    
    # # Imprime linha onde título == 1
    # pos = comprimento[np.where(título == 0)[0][0]]
    # valor = título[np.where(título == 0)[0][0]]
    # plt.scatter(pos,valor, s = 8, color='red')
    # plt.text(pos - 0.2, 0, f'L = {pos*1e3:.2f}mm')
    
    # Gráfico da distribuição de temperaturas   
    Y = np.linspace(0.5, N_v+0.5, N_v+1)
    X = np.linspace(0, L_tubo, N_elemento + 1)
    # Segundo_passe = np.flip(T_f_in[Passes[0]:Passes[0]+Passes[1],:],axis=1)
    # T_f_in[Passes[0]:Passes[0]+Passes[1],:] = Segundo_passe
    # T_f_in = np.flip(T_f_in,axis=0)
    plt.figure(dpi=200)
    plot = plt.pcolormesh(X, Y, T_f_in-273.15, shading='auto')
    plt.colorbar(plot)
    plt.xlabel("Length [m]")
    plt.ylabel("Tube number")
    plt.title('Temperature x Length')
    for line in range(1,N_v+1):
        plt.axhline(y = line-0.5, color = 'black', linestyle='-.', linewidth = 0.3)
    plt.axhline(y = 25.5, color = 'red', linestyle='--')
    plt.axhline(y = 40.5, color = 'red', linestyle='--')
    
    # Plota distribuição do título
    A_elemento = 0.005480061120630762
    plt.figure(dpi=200)
    plt.scatter(comp_HTC, 1 / (HTC / A_elemento), color='black', s = 6)
    plt.scatter(comp_HTC, 1 / H_f, color='red', s = 6)
    plt.scatter(comp_HTC, 1 / H_ar, color='blue', s = 6)
    plt.title(f"$N_{{elementos}}$ = {N_elemento}")
    plt.ylabel("U [W/$m^2$K]")
    plt.xlabel("Comprimento [m]")
    for linha in linhas:   
        plt.axvline(x = linha, color = 'black', linestyle='-.', linewidth = 0.4)    
    plt.xlim([0,L_tubo*len(Passes)])

  
N_elementos_total = [i * sum(Passes) for i in N_elementos]
plt.figure(dpi=200)
plt.plot(N_elementos, tempo, marker='.', color = 'black')
plt.xlabel("$N_{{elementos}}$ / tubo")
plt.ylabel("Tempo [s]")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
axisy2 = plt.gca().twiny()
axisy2.plot(N_elementos_total,tempo, marker = '.', color='black')
plt.xlabel("$N_{{elementos}}$ Total")

plt.figure(dpi=200)
plt.plot(N_elementos[1:],dif, marker = '.', color='black')
plt.ylabel("Variação")
plt.xlabel("$N_{{elementos}}$ / tubo")
axisy2 = plt.gca().twiny()
axisy2.plot(N_elementos_total[1:],dif,marker = '.', color='black')
plt.xlabel("$N_{{elementos}}$ Total")





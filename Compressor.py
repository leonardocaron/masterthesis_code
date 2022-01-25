# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:42:08 2021

@author: Usuário
"""

import scipy.constants
import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def compressor_sierra_pol(N, T_e, T_c):
    x_1 = N
    x_2 = scipy.constants.convert_temperature(T_e,'Celsius','Fahrenheit')
    x_3 = scipy.constants.convert_temperature(T_c,'Celsius','Fahrenheit')
      
    # Vazão mássica    
    C1 = -2.034164E+02
    C2 = 2.796228E-02
    C3 = -2.825652E-06
    C4 = 2.498261E-10
    C5 = 1.281964E+00
    C6 = 7.525860E-03
    C7 = 1.188066E-04
    C8 = 4.763349E+00
    C9 = -3.839735E-02
    C10 = 9.837328E-05
    C11 = 1.433789E-05
    C12 = 4.227007E-10
    C13 = 2.606284E-08
    C14 = -7.249366E-08
    C15 = -2.738471E-04
    C16 = -7.651051E-05
    C17 = -2.882155E-02
    C18 = -5.668466E-08
    C19 = 1.314290E-06
    C20 = -3.070176E-09
    C21 = 4.219426E-07
    C22 = -1.108327E-04
    C23 = 1.413719E-04
    
    m = C1 + C2 * x_1 + C3 * x_1 ** 2 + C4 * x_1 ** 3 + C5 * x_2 + C6 * x_2 ** 2 + C7 * x_2 ** 3 + \
        C8 * x_3 + C9 * x_3 ** 2 + C10 * x_3 ** 3 + C11 * x_1 * x_2 * x_3 + C12 * x_1 ** 2 * x_2 * x_3 + \
            C13 * x_1 * x_2 ** 2 * x_3 + C14 * x_1 * x_2 * x_3 ** 2 + C15 * x_1 * x_2 + C16 * x_1 * x_3 + \
                C17 * x_2 * x_3 + C18 * x_1 ** 2 * x_2 + C19 * x_1 * x_2 ** 2 + C20 * x_1 ** 2 * x_3 + \
                    C21 * x_1 * x_3 ** 2 + C22 * x_2 ** 2 * x_3 + C23 * x_2 * x_3 ** 2
    m = m * 0.4535924
    
    # Trabalho
    C1 = -2.217077E+3
    C2 = 5.752299E-1
    C3 = -3.694308E-5
    C4 = 2.206452E-9
    C5 = 1.584974E+1
    C6 = 8.844898E-1
    C7 = -2.792083E-3
    C8 = 3.5184E+1
    C9 = -1.202478E-1
    C10 = -1.717872E-4
    C11 = 1.85128E-5
    C12 = 5.989812E-9
    C13 = 2.117815E-6
    C14 = -6.424219E-7
    C15 = 3.943979E-3
    C16 = -6.765075E-3
    C17 = -6.092957E-1
    C18 = -3.553804E-7
    C19 = -2.177671E-4
    C20 = 1.629147E-7
    C21 = 3.125445E-5
    C22 = -6.715211E-3
    C23 = 4.426135E-3
    
    W = C1 + C2 * x_1 + C3 * x_1 ** 2 + C4 * x_1 ** 3 + C5 * x_2 + C6 * x_2 ** 2 + C7 * x_2 ** 3 + \
         C8 * x_3 + C9 * x_3 ** 2 + C10 * x_3 ** 3 + C11 * x_1 * x_2 * x_3 + C12 * x_1 ** 2 * x_2 * x_3 + \
             C13 * x_1 * x_2 ** 2 * x_3 + C14 * x_1 * x_2 * x_3 ** 2 + C15 * x_1 * x_2 + C16 * x_1 * x_3 + \
                 C17 * x_2 * x_3 + C18 * x_1 ** 2 * x_2 + C19 * x_1 * x_2 ** 2 + C20 * x_1 ** 2 * x_3 + \
                     C21 * x_1 * x_3 ** 2 + C22 * x_2 ** 2 * x_3 + C23 * x_2 * x_3 ** 2
  
    return m, W

def compressor_sierra_eq(P_dis, P_suc, k, N, T_suc):
    # Parâmetros da regressão da vazão
    N_ref = 4500
    V_des = 16.1e-6
    b_1 = 1.1269516299373805
    b_2 = -1003.5201149956131
    dp = -109876.04679215071
    d_1 = 0.9375090348865804
    d_2 = 0.1108837507425683
    d_3 = -0.047835425987342806
      
    eta_v_ref = b_1 + b_2 * ((P_dis / (P_suc * (1 - dp))) ** (1 / k))
    eta_v = (d_1 + d_2 * (N / N_ref) + d_3 * (N / N_ref) ** 2) * eta_v_ref
    rho_suc = CP.PropsSI('D', 'T', T_suc, 'P', P_suc, 'R134a')
    m_dot = (N / 60) * V_des * eta_v * rho_suc
    
    # Parâmetros da regressão do consumo
    N_ref = 4500
    V_des = 16.1e-6 
    a_1 = 0.022028948268757353
    a_2 = 0.34505044002501517
    a_3 = -757099.7486305697
    W_loss = 283.2553819205746
    e_1 = 1.3340544620954427
    e_2 = -0.8446227650167992
    e_3 = 0.5092285932053934
    
    V_suc_ref = V_des * eta_v_ref * N_ref
    W_ref = P_suc * V_suc_ref * a_1 * ((P_dis / P_suc) ** (a_2 + (k - 1) / k) + a_3 / P_dis) + W_loss
    V_suc = V_des * eta_v * N
    W = W_ref * (V_suc / V_suc_ref) * (e_1 + e_2 * (N / N_ref) + e_3 * (N / N_ref) ** 2)

    
    return m_dot, W

def ajuste_eta_v():
    
    Dados_ref = {
        "eta_v": [], 
        "k": [],
        "P_d": [],
        "P_s": [],
        }
    
    N_ref = 4500
    Rots = np.array([N_ref], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    V_des = 16.1e-6
    T_suc = 18.33 + 273.15
    
    for Rot in Rots:
       
        for T_c in T_cs:
            
            for T_e in T_es:
                
                m_dot, W_dot = compressor_sierra_pol(Rot, T_e, T_c)
                
                #Computo eficiencia volumétrica
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                P_dis = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                rho_suc = CP.PropsSI("D", 'P', P_suc, 'T', T_suc, 'R134a')
                m_dot_ideal = rho_suc * (Rot / 60) * V_des
                eta_v = (m_dot / 3600) / m_dot_ideal
                
                Dados_ref["P_d"].append(CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a'))
                Dados_ref["P_s"].append(CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a'))
                Dados_ref["eta_v"].append(eta_v)
                Dados_ref["k"].append(k)
  
    parametros = curve_fit(eq_eta_v_ref, [Dados_ref["P_d"], Dados_ref["P_s"], Dados_ref["k"]], Dados_ref["eta_v"], bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 1]), p0 = [1, 0.02, 0])
    b_0 = parametros[0][0]
    b_1 = parametros[0][1]
    b_2 = parametros[0][2]   

    # Ajuste para demais rotações
    Rots = np.array([3500, 4000,5000], dtype='f')    
    Dados = {
        "eta_v": [], 
        "Rotação": [],
        "eta_v_ref": [],
        "Rotação_ref": [],
        }
    
    for Rot in Rots:
        
        for T_c in T_cs:
            
            for T_e in T_es:
                
                # Propriedades fixas
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                rho_suc = CP.PropsSI("D", 'P', P_suc, 'T', T_suc, 'R134a')
                
                # Dados de referência
                m_dot_ref, W_dot_ref = compressor_sierra_pol(N_ref, T_e, T_c)
                m_dot_ideal_ref = rho_suc * (N_ref / 60) * V_des
                eta_v_ref = (m_dot_ref / 3600) / m_dot_ideal_ref
                
                # Dados na rotação pedida
                m_dot, W_dot = compressor_sierra_pol(Rot, T_e, T_c)
                m_dot_ideal = rho_suc * (Rot / 60) * V_des
                eta_v = (m_dot / 3600) / m_dot_ideal
                               
                Dados["eta_v"].append(eta_v)
                Dados["Rotação"].append(Rot)
                Dados["eta_v_ref"].append(eta_v_ref)
                Dados["Rotação_ref"].append(N_ref)
            
    parametros = curve_fit(eq_eta_v, [Dados["eta_v_ref"], Dados["Rotação"], Dados["Rotação_ref"]], Dados["eta_v"])
    d_0 = parametros[0][0]
    d_1 = parametros[0][1]
    d_2 = parametros[0][2]
    
    # Gráficos
    Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    
    Ajustado = {
        "Rotação": [], 
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Vazão2": [],
        }
    
    Calculado = {
        "Rotação": [], 
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        }
    
    N_ref = 4500
    
    for Rot in Rots:
        
        for T_c in T_cs:
            
            for T_e in T_es:
                
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                P_dis = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                
                eta_v_ref = eq_eta_v_ref([P_dis, P_suc, k], b_0, b_1, b_2)
                eta_v_ajustado = eq_eta_v([eta_v_ref, Rot, N_ref], d_0, d_1, d_2)
                
                rho_suc = CP.PropsSI("D", 'P', P_suc, 'T', T_suc, 'R134a')
                m_ideal = rho_suc * (Rot / 60) * V_des
                m_ajustado = eta_v_ajustado * m_ideal * 3600
                
                Ajustado["Rotação"].append(Rot)
                Ajustado["Cond"].append(T_c)
                Ajustado["Evap"].append(T_e)
                Ajustado["Vazão"].append(m_ajustado)
       
                m_dot_calculado, W_calculado = compressor_sierra_pol(Rot, T_e, T_c)
                            
                Calculado["Rotação"].append(Rot)
                Calculado["Cond"].append(T_c)
                Calculado["Evap"].append(T_e)
                Calculado["Vazão"].append(m_dot_calculado)
                
                # m_ajustado2, W_ajustado2 = compressor_sierra_eq(P_dis, P_suc, k, Rot, 18.33 + 273.15)
                # Ajustado["Vazão2"].append(m_ajustado2*3600)
    
    inicio = 0
    fim = len(T_es)
    for figure, Rot in enumerate(Rots):
        plt.figure(figure)
        plt.figure(dpi=300)
        plt.title(f"Rotação {Rot:.0f} rpm - Abordagem Li (2013)")
        plt.xlabel("$T_{evap}$ [°C]")
        plt.ylabel("$\dot{m}$ [kg/h]")
        
        for T_c in T_cs:
            plt.scatter(T_es, Calculado["Vazão"][inicio:fim])
            plt.plot(T_es, Ajustado["Vazão"][inicio:fim], label = f'${{T_c}}$: {T_c}')
            plt.legend()
            inicio = fim
            fim += len(T_es)
            
    plt.figure(figure + 1)
    
    #Gráfico de erros
    plt.figure(dpi = 200)
    plt.scatter(Calculado["Vazão"], Ajustado["Vazão"])
    plt.xlim([Calculado["Vazão"][0]*0.90,Calculado["Vazão"][-1]*1.05])
    plt.ylim([Calculado["Vazão"][0]*0.90,Calculado["Vazão"][-1]*1.05])
    
    # Erros
    Erro = [0, 1000000]
    Erro_10 = [0, 1000000 * 1.05]
    Erro_10_ = [0, 1000000 * 0.95]
    plt.plot(Erro, Erro, color = 'black',  linewidth = 0.5, linestyle = ':', label = "0%")
    plt.plot(Erro, Erro_10, color = 'black', linewidth = 0.5, linestyle = '-.' , label = "+5%")
    plt.plot(Erro, Erro_10_, color = 'black', linewidth = 0.5, linestyle = '--', label = "-5%")
    plt.legend()
    plt.title("Erros - Abordagem Li (2019)")
    plt.ylabel("$\dot{m}_{ajustado}$ [kg/h]")
    plt.xlabel("$\dot{m}_{catálogo}$ [kg/h]")
            
            
    return b_0, b_1, b_2, d_0, d_1, d_2
            
def eq_eta_v_ref(X, b_1, b_2, dp):
    P_dis, P_suc, k = X
    eta_v = b_1 + b_2 * ((P_dis / (P_suc * (1 - dp))) ** (1 / k))
    return eta_v

def eq_eta_v(X, d_1, d_2, d_3):
    eta_v_ref, N, N_ref = X
    eta_v = (d_1 + d_2 * (N / N_ref) + d_3 * (N / N_ref) ** 2) * eta_v_ref
    return eta_v

def ajuste_vazao():
   
    Dados = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [],    
        "P_d": [],
        "P_s": []
        }
    
    Ajustado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [],      
        }
    
    Calculado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [],      
        }    
    
    Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    
    for Rot in Rots:       
        for T_c in T_cs:           
            for T_e in T_es:              
                m_dot, W_dot = compressor_sierra_pol(Rot, T_e, T_c)
                Dados["Vazão"].append(m_dot)
                Dados["Rotação"].append(Rot)
                Dados["Cond"].append(T_c)
                Dados["Evap"].append(T_e)
                Dados["Consumo"].append(W_dot)
                Dados["P_d"].append(CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a'))
                Dados["P_s"].append(CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a'))
 
    Dados["T_s"] = [18.33 + 273.15 for T in Dados["Cond"]]
 
    # Ajuste da curva
    parametros = curve_fit(eq_vazao, [Dados["T_s"], Dados["Rotação"], Dados["P_s"], Dados["P_d"]], Dados["Vazão"])
    b_0 = parametros[0][0]
    b_1 = parametros[0][1]
    b_2 = parametros[0][2]

    # # Dados para os gráficos
    # Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    # T_cs = np.array([50, 55, 60], dtype='f')
    # T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    
    # for Rot in Rots:        
    #     for T_c in T_cs:           
    #         for T_e in T_es:
                
    #             # Curva ajustada
    #             P_d = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
    #             P_s = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
    #             vaz_ajustado = vazao([18.3+273.15, Rot, P_s, P_d], b_0,b_1,b_2)
    #             Ajustado["Vazão"].append(vaz_ajustado)
    #             Ajustado["Rotação"].append(Rot)
    #             Ajustado["Cond"].append(T_c)
    #             Ajustado["Evap"].append(T_e)

    #             # Valores catálogo               
    #             vaz_calculado, W_calculado = compressor_sierra(Rot, T_e, T_c)
    #             Calculado["Vazão"].append(vaz_calculado)
    #             Calculado["Rotação"].append(Rot)
    #             Calculado["Cond"].append(T_c)
    #             Calculado["Evap"].append(T_e)
   
    # # Gráfico das curvas ajustadas
    # inicio = 0
    # fim = len(T_es)
    # for figure, Rot in enumerate(Rots):
    #     plt.figure(figure)
    #     plt.figure(dpi=300)
    #     plt.title(f"Rotação {Rot:.0f} rpm - Abordagem Santos, Ronzoni e Hermes (2019)")
    #     plt.xlabel("$T_{evap}$ [°C]")
    #     plt.ylabel("$\dot{m}$ [kg/h]")
        
    #     for T_c in T_cs:
    #         plt.scatter(T_es, Calculado["Vazão"][inicio:fim])
    #         plt.plot(T_es, Ajustado["Vazão"][inicio:fim], label = f'${{T_c}}$: {T_c}')
    #         plt.legend()
    #         inicio = fim
    #         fim += len(T_es)
            
    # # Gráfico dos erros
    # plt.figure(figure + 1)
    # plt.figure(dpi = 200)
    # plt.scatter(Calculado["Vazão"], Ajustado["Vazão"])
    # plt.xlim([Calculado["Vazão"][0]*0.90,Calculado["Vazão"][-1]*1.05])
    # plt.ylim([Calculado["Vazão"][0]*0.90,Calculado["Vazão"][-1]*1.05])
    
    # # Erros
    # Erro = [0, 1000000]
    # Erro_10 = [0, 1000000 * 1.05]
    # Erro_10_ = [0, 1000000 * 0.95]
    # plt.plot(Erro, Erro, color = 'black',  linewidth = 0.5, linestyle = ':', label = "0%")
    # plt.plot(Erro, Erro_10, color = 'black', linewidth = 0.5, linestyle = '-.' , label = "+5%")
    # plt.plot(Erro, Erro_10_, color = 'black', linewidth = 0.5, linestyle = '--', label = "-5%")
    # plt.legend()
    # plt.title("Erros - Abordagem Santos, Ronzoni e Hermes (2019)")
    # plt.ylabel("$\dot{m}_{ajustado}$ [kg/h]")
    # plt.xlabel("$\dot{m}_{catálogo}$ [kg/h]")
           
    return b_0, b_1, b_2

def eq_vazao(X, b_0, b_1, b_2):
    T_s,N,P_s,P_d = X
    m_dot = (P_s * N / (T_s)) * (b_0 - b_1 * ((P_d/P_s) ** b_2 - 1))    
    return m_dot
    
def ajuste_potencia():

    Dados = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [], 
        "P_d": [],
        "P_s": [],
        "T_s": []
        }
    
    Ajustado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [],      
        }
    
    Calculado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "Vazão": [],
        "Consumo": [],      
        }        
    
    Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    
    b_0, b_1, b_2 = ajuste_vazao()  
    
    for Rot in Rots:      
        for T_c in T_cs:           
            for T_e in T_es:               
                P_d = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                P_s = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                T_s = 18.33 + 273.15                
                m_dot, W_dot = compressor_sierra_pol(Rot, T_e, T_c)
                m_dot_ajustado = eq_vazao([T_s, Rot, P_s, P_d], b_0, b_1, b_2)
                              
                Dados["Vazão"].append(m_dot_ajustado)
                Dados["Rotação"].append(Rot)
                Dados["Cond"].append(T_c)
                Dados["Evap"].append(T_e)
                Dados["Consumo"].append(W_dot)  
                Dados["P_d"].append(P_d)
                Dados["P_s"].append(P_s)
                Dados["T_s"].append(18.33 + 273.15)
                 
    parametros = curve_fit(eq_potencia, [Dados["T_s"], Dados["Rotação"], Dados["P_s"], Dados["P_d"], Dados["Vazão"]], Dados["Consumo"])
    a_0 = parametros[0][0]
    a_1 = parametros[0][1]
    a_2 = parametros[0][2]  
    
    # Gráficos
    Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')  
    
    for Rot in Rots:       
        for T_c in T_cs:           
            for T_e in T_es:
                
                # Valores catálogo
                vaz, W_calculado = compressor_sierra_pol(Rot, T_e, T_c)
                Calculado["Consumo"].append(W_calculado)
                
                # Valores ajustados
                P_d = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                P_s = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                T_s = 18.33 + 273.15
                vaz_ajustado = eq_vazao([T_s, Rot, P_s, P_d], b_0, b_1, b_2)
                W_ajustado = eq_potencia([T_s, Rot, P_s, P_d, vaz_ajustado], a_0,a_1, a_2)              
                Ajustado["Consumo"].append(W_ajustado)

    inicio = 0
    fim = len(T_es)
    for figure, Rot in enumerate(Rots):
        plt.figure(figure)
        plt.figure(dpi=300)
        plt.title(f"Rotação {Rot:.0f} rpm - Abordagem Santos, Ronzoni e Hermes (2019)")
        plt.xlabel("$T_{evap}$ [°C]")
        plt.ylabel("$\dot{W}$ [W]")
        
        for T_c in T_cs:
            plt.scatter(T_es, Calculado["Consumo"][inicio:fim])
            plt.plot(T_es, Ajustado["Consumo"][inicio:fim], label = f'${{T_c}}$: {T_c}')
            plt.legend()
            inicio = fim
            fim += len(T_es)
            
    plt.figure(figure + 1)
    #Gráfico de erros
    plt.figure(dpi = 200)
    plt.scatter(Calculado["Consumo"], Ajustado["Consumo"])
    plt.xlim([Calculado["Consumo"][0]*0.95, Calculado["Consumo"][-1]*1.05])
    plt.ylim([Calculado["Consumo"][0]*0.95, Calculado["Consumo"][-1]*1.05])
    
    # Erros
    Erro = [0, 1000000]
    Erro_10 = [0, 1000000 * 1.05]
    Erro_10_ = [0, 1000000 * 0.95]
    plt.plot(Erro, Erro, color = 'black',  linewidth = 0.5, linestyle = ':', label = "0%")
    plt.plot(Erro, Erro_10, color = 'black', linewidth = 0.5, linestyle = '-.' , label = "+5%")
    plt.plot(Erro, Erro_10_, color = 'black', linewidth = 0.5, linestyle = '--', label = "-5%")
    plt.legend()
    plt.title("Erros - Abordagem Santos, Ronzoni e Hermes (2019)")
    plt.ylabel("$W_{ajustado}$ [W]")
    plt.xlabel("$W_{catálogo}$ [W]")
           
    return 

def eq_potencia(X, a_0, a_1, a_2):
    T_s,N,P_s,P_d, m_dot = X    
    W_dot = m_dot * (a_0 * T_s * ((P_d/P_s) ** a_1 -1) + a_2)       
    return W_dot

        
def ajuste_potencia_li():
    
    Dados_ref = {
        "P_d": [],
        "P_s": [],
        "V_suc": [],
        "Potência": [],
        "k": [],
        }
    
    N_ref = 4500
    Rots = np.array([N_ref], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f')
    V_des = 16.1e-6
    T_suc = 18.33 + 273.15
    
    b_0, b_1, b_2, d_1, d_2, d_3 = ajuste_eta_v()
    
    for Rot in Rots:
        
        for T_c in T_cs:
            
            for T_e in T_es:
                
                vaz, W = compressor_sierra_pol(Rot, T_e, T_c)
                
                #Computo eficiencia volumétrica
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                P_dis = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                eta_v = eq_eta_v_ref([P_dis, P_suc, k], b_0, b_1, b_2)
                V_suc = eta_v * V_des * Rot       
                vaz, W = compressor_sierra_pol(Rot, T_e, T_c)
                
                Dados_ref["P_d"].append(P_dis)
                Dados_ref["P_s"].append(P_suc)
                Dados_ref["V_suc"].append(V_suc)
                Dados_ref["Potência"].append(W)
                Dados_ref["k"].append(k)
    
    parametros = curve_fit(potencia_ref_li, [Dados_ref["P_d"], Dados_ref["P_s"], Dados_ref["k"], Dados_ref["V_suc"]], Dados_ref["Potência"])
    a_1 = parametros[0][0]
    a_2 = parametros[0][1]
    a_3 = parametros[0][2]
    W_loss = parametros[0][3]

    # Ajuste demais rotações
    Rots = np.array([3500, 4000, 5000], dtype='f')    
    Dados = {
        "Rotação": [],
        "V_suc": [],
        "Potência": [],
        "Rotação_ref": [],
        "V_suc_ref": [],
        "Potência_ref": [],
        "P_d": [],
        "P_s": [], 
        "k":[],
        }

    for Rot in Rots:
        
        for T_c in T_cs:
            
            for T_e in T_es:
                
                # Propriedades fixas
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                P_dis = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                rho_suc = CP.PropsSI("D", 'P', P_suc, 'T', T_suc, 'R134a')
                
                # Dados de referência
                vaz_ref, W_ref = compressor_sierra_pol(N_ref, T_e, T_c)
                eta_v_referencia = eq_eta_v_ref([P_dis, P_suc, k], b_0, b_1, b_2)
                V_suc_ref = V_des * N_ref * eta_v_referencia
                
                # Dados na rotação correta
                vaz, W = compressor_sierra_pol(Rot, T_e, T_c)
                eta_v = eq_eta_v([eta_v_referencia, Rot, N_ref], d_1, d_2, d_3)
                V_suc = V_des * Rot * eta_v
                
                Dados["Rotação"].append(Rot)
                Dados["V_suc"].append(V_suc)
                Dados["Potência"].append(W)
                Dados["Rotação_ref"].append(N_ref)
                Dados["V_suc_ref"].append(V_suc_ref)
                Dados["Potência_ref"].append(W_ref)
                Dados["P_d"].append(P_dis)
                Dados["P_s"].append(P_suc)
                Dados["k"].append(k)
   
    parametros = curve_fit(potencia_rot_li, [Dados["Potência_ref"], Dados["V_suc_ref"], Dados["Rotação_ref"], Dados["V_suc"], Dados["Rotação"]], Dados["Potência"])
    e_1 = parametros[0][0]
    e_2 = parametros[0][1]
    e_3 = parametros[0][2]


    # Gráficos
    Rots = np.array([3500, 4000, 4500, 5000], dtype='f')
    T_cs = np.array([50, 55, 60], dtype='f')
    T_es = np.array([0, 2.5, 5, 7.5, 10], dtype='f') 
    
    Ajustado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "W": [],
        "W2": [],
        }
    Calculado = {
        "Rotação": [],
        "Cond": [],
        "Evap": [],
        "W": [],
        }
    N_ref = 4500
    
    for Rot in Rots:
        
        for T_c in T_cs:
            
            for T_e in T_es:
                
                P_suc = CP.PropsSI('P', 'T', T_e + 273.15, 'Q', 1, 'R134a')
                P_dis = CP.PropsSI('P', 'T', T_c + 273.15, 'Q', 1, 'R134a')
                k = CP.PropsSI('CPMASS', 'P', P_suc, 'T', T_suc, 'R134a') / CP.PropsSI('CVMASS', 'P', P_suc, 'T', T_suc, 'R134a')
                
                # Dados de referência
                eta_v_referencia = eq_eta_v_ref([P_dis, P_suc, k], b_0, b_1, b_2)
                V_suc_ref = V_des * N_ref * eta_v_referencia
                W_ref = potencia_ref_li([P_dis, P_suc, k, V_suc_ref], a_1, a_2, a_3, W_loss)
                
                # Dados na rotação correta
                eta_v = eq_eta_v([eta_v_referencia, Rot, N_ref], d_1, d_2, d_3)
                V_suc = V_des * Rot * eta_v
                W_ajustado = potencia_rot_li([W_ref, V_suc_ref, N_ref, V_suc, Rot], e_1, e_2, e_3)
                
                Ajustado["Rotação"].append(Rot)
                Ajustado["Cond"].append(T_c)
                Ajustado["Evap"].append(T_e)
                Ajustado["W"].append(W_ajustado)
                                
                vaz_calculado, W_calculado = compressor_sierra_pol(Rot, T_e, T_c)
                
                Calculado["Rotação"].append(Rot)
                Calculado["Cond"].append(T_c)
                Calculado["Evap"].append(T_e)
                Calculado["W"].append(W_calculado)
                
                m_ajustado2, W_ajustado2 = compressor_sierra_eq(P_dis, P_suc, k, Rot, 18.33 + 273.15)
                Ajustado["W2"].append(W_ajustado2)
    
    inicio = 0
    fim = len(T_es)
    for figure, Rot in enumerate(Rots):
        plt.figure(figure)
        plt.figure(dpi=300)
        plt.title(f"Rotação {Rot:.0f} rpm - Abordagem Li (2013)")
        plt.xlabel("$T_{evap}$ [°C]")
        plt.ylabel("$\dot{W}$ [W]")
        
        for T_c in T_cs:
            plt.scatter(T_es, Calculado["W"][inicio:fim])
            plt.plot(T_es, Ajustado["W"][inicio:fim], label = f'${{T_c}}$: {T_c}')
            plt.legend()
            inicio = fim
            fim += len(T_es)
            
    plt.figure(figure + 1)
    #Gráfico de erros
    plt.figure(dpi = 200)
    plt.scatter(Calculado["W"], Ajustado["W"])
    plt.xlim([Calculado["W"][0]*0.95,Calculado["W"][-1]*1.05])
    plt.ylim([Calculado["W"][0]*0.95,Calculado["W"][-1]*1.05])
    
    # Erros
    Erro = [0, 1000000]
    Erro_10 = [0, 1000000 * 1.05]
    Erro_10_ = [0, 1000000 * 0.95]
    plt.plot(Erro, Erro, color = 'black',  linewidth = 0.5, linestyle = ':', label = "0%")
    plt.plot(Erro, Erro_10, color = 'black', linewidth = 0.5, linestyle = '-.' , label = "+5%")
    plt.plot(Erro, Erro_10_, color = 'black', linewidth = 0.5, linestyle = '--', label = "-5%")
    plt.legend()
    plt.title("Erros - Abordagem Li (2013)")
    plt.ylabel("$W_{ajustado}$ [W]")
    plt.xlabel("$W_{catálogo}$ [W]")
    
def potencia_ref_li(X, a_1, a_2, a_3, W_loss):
    P_dis, P_suc, k, V_suc = X
    W = P_suc * V_suc * a_1 * ((P_dis / P_suc) ** (a_2 + (k - 1) / k) + a_3 / P_dis) + W_loss
    return W

def potencia_rot_li(X, e_1, e_2, e_3):
    W_ref, V_suc_ref, N_ref, V_suc, N = X
    W = W_ref * (V_suc / V_suc_ref) * (e_1 + e_2 * (N / N_ref) + e_3 * (N / N_ref) ** 2)
    return W


if __name__ == '__main__':
   #ajuste_vazao()
   #ajuste_eta_v()
   ajuste_potencia()
   ajuste_potencia_li()
    
    
    
    
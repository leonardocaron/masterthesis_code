# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:36:35 2021

@author: Usuário
"""

import CoolProp
import numpy as np
import matplotlib.pyplot as plt

def h_bifasico(G_f, fluido, d_i_tubo, P_f_in, x_f_in, HEOS_fluido, BICUBIC_fluido):
    # Equação de Shah 1979
    # Calculo h_lo
    # k_f_l = CP.PropsSI('CONDUCTIVITY', 'P', P_f_in, 'Q', 0, fluido)
    # Pr_f_l = CP.PropsSI('PRANDTL', 'P', P_f_in, 'Q', 0, fluido)
    # mu_f_l = CP.PropsSI('VISCOSITY', 'P', P_f_in, 'Q', 0, fluido)
    # Re_d_f_lo = G_f*d_i_tubo/mu_f_l
    # P_crit = CP.PropsSI("PCRIT", fluido)
    # P_r = P_f_in / P_crit

    # # coeficiente de transferência para o fluido
    # h_lo = (k_f_l / d_i_tubo) * 0.0265 * (Re_d_f_lo**(4/5)) * (Pr_f_l)**(0.3)
    # h_f = h_lo * ((1 - x_f_in)**0.8 + (3.8 * x_f_in**0.76 * (1 - x_f_in)**0.04) / ((P_f_in / P_crit)**0.38))

    # Shah microcanal
    Props_vapor = BICUBIC_fluido
    Props_vapor.update(CoolProp.PQ_INPUTS, P_f_in, 1)
    rho_f_g = Props_vapor.rhomass()
    mu_f_g = Props_vapor.viscosity()
    P_crit = Props_vapor.p_critical()
    P_r = P_f_in / P_crit
        
    Props_liquido = BICUBIC_fluido
    Props_liquido.update(CoolProp.PQ_INPUTS, P_f_in, 0)   
    rho_f_l = Props_liquido.rhomass()
    mu_f_l = Props_liquido.viscosity()
    k_f_l = Props_liquido.conductivity()
    Pr_f_l = Props_liquido.Prandtl()
    
    # Props = HEOS_fluido
    # Props.update(CoolProp.PQ_INPUTS, P_f_in, 0)
    # sigma = Props.surface_tension()
    
    sigma = surface_tension_global
   
    # # Números adimensionais 
    We_gt = G_f ** 2 * d_i_tubo / (rho_f_g * sigma)
    Re_lo = G_f * (1 - x_f_in) * d_i_tubo / mu_f_l
    Re_lt = G_f * d_i_tubo / mu_f_l
    g = 9.80665
    J_g = (x_f_in * G_f) / ((g * d_i_tubo * rho_f_g * (rho_f_l - rho_f_g)) ** 0.5)
    Z = (1 / x_f_in - 1) ** 0.8 * P_r ** 0.4
    
    # # # Coeficientes de convecção
    h_LT = 0.023 * (Re_lt ** 0.8) * (Pr_f_l ** 0.4) * (k_f_l / d_i_tubo)
    h_Nu = 1.32 * (Re_lo ** (-1/3)) * ((rho_f_l * (rho_f_l - rho_f_g) * g * k_f_l ** 3 / (mu_f_l ** 2)) ** (1 / 3))
    h_I = h_LT * (1 + 1.128 * x_f_in ** 0.817 * (rho_f_l / rho_f_g) ** 0.3685 * (mu_f_l / mu_f_g) ** 0.2363 * (1 - mu_f_g / mu_f_l) ** 2.144 * (Pr_f_l ** (-0.1)))
      
    Regime_1 = We_gt >= 100 and J_g >= 0.98 * (Z + 0.263) ** (-0.62)
    Regime_3 = We_gt > 20 and J_g <= 0.95 * (1.254 + 2.27 * Z ** 1.249 ) ** (-1)
    Regime_2 = not Regime_1 and not Regime_3

    if Regime_1:
        h_tp = h_I
    elif Regime_2:
        h_tp = h_I + h_Nu
    elif Regime_3:
        h_tp = h_Nu
       
    return h_tp


fluido = 'R134a'
G_f = 217
P_f_inlet = 1027542.586376925
d_h_canal = 0.0007

HEOS_fluido = CoolProp.AbstractState("HEOS", fluido)
BICUBIC_fluido = CoolProp.AbstractState("BICUBIC", fluido)
HEOS_Ar = CoolProp.AbstractState("HEOS", "Air")

HEOS_fluido.update(CoolProp.PQ_INPUTS, P_f_inlet, 1)
global surface_tension_global
surface_tension_global = HEOS_fluido.surface_tension()

xs = np.arange(0.02, 1.01, 0.0001)
h_tp = []
for x in xs:
    h_tp.append(h_bifasico(G_f, fluido, d_h_canal, P_f_inlet, x, HEOS_fluido, BICUBIC_fluido))
    
plt.figure(dpi=300)
plt.plot(xs,h_tp, color = 'black')
plt.ylabel("h [W/$m^2$K]")
plt.xlabel("x [-]")
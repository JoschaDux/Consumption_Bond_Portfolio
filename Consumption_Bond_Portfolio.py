#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:28:47 2025

@author: joschaduchscherer
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# Define model parameters
class Model:
    def __init__(self):
        self.lam = -0.05 # Market-Price of Risk
        self.kappa = 0.15 # Mean-Reversion Speed
        self.theta = 0.04 # Mean-Reversion Level
        self.beta = 0.015 # Volatility r
        self.gam = 10 # Relative Risk-Aversion
        self.psi_list = [0.1, 0.5, 1.0, 1.5] # EIS
        self.delta = 0.015 # Time-Preference Rate
        self.T_bond = 100 # Maturity Bond
        self.eps = 2 # Bequest-Motive

#Define grid parameters
class Grid:
    def __init__(self):
        self.rmin = -0.05
        self.rmax = 0.3
        self.tmin = 0
        self.tmax = 50
        self.Nr = 100
        self.Nt = 1000000
        self.dt = (self.tmax - self.tmin) / self.Nt
        self.dr = (self.rmax - self.rmin) / self.Nr
        self.nr = np.arange(self.Nr + 1)
        self.nt = np.arange(self.Nt + 1)
        self.r = self.rmin + self.dr* self.nr

model = Model()
grid = Grid()

#Define policy function
def policy(g_ns, psi):
    # Calculate numerical derivative
    g_ns_r = (np.roll(g_ns, -1) - np.roll(g_ns, 1)) / (2 * grid.dr)
    
    # Linear extrapolation at lower bound
    slope_lb = (g_ns_r[1] - g_ns_r[2]) / (grid.r[1] - grid.r[2])
    intercept_lb = g_ns_r[1] - slope_lb * grid.r[1]
    g_ns_r[0] = slope_lb*grid.r[0]+intercept_lb
    
    # Linear extrapolation at upper bound
    slope_ub = (g_ns_r[-2] - g_ns_r[-3]) / (grid.r[-2] - grid.r[-3])
    intercept_ub = g_ns_r[-2] - slope_ub * grid.r[-2]
    g_ns_r[-1] = slope_ub*grid.r[-1]+intercept_ub
    
    # Calculate optimal consumption wealth-ratio
    if psi != 1:
        theta_pref = (1-model.gam)/(1-1/psi)
        cw = (1/model.delta*g_ns**(1/theta_pref))**(-psi)
    else:
        cw = model.delta*np.ones(grid.Nr + 1)
    
    # Calculate optimal portfolio share
    D = -model.lam/(model.gam*model.beta)-1/model.gam*g_ns_r/g_ns
    
    # Linear extrapolation for pi at lower bound
    slope_lb = (D[1]-D[2])/ (grid.r[1] - grid.r[2])
    intercept_lb= D[1] - slope_lb* grid.r[1]
    D[0] = slope_lb * grid.r[0] + intercept_lb
    
    # Linear extrapolation of pi at upper bound
    slope_ub = (D[-2] - D[-3]) / (grid.r[-2] - grid.r[-3])
    intercept_ub = D[-2] - slope_ub * grid.r[-2]
    D[-1] = slope_ub* grid.r[-1] + intercept_ub
    
    return cw, D

#Define function for the coefficients of the finite difference method
def coefficients(cw, D):
    
    drift_x = grid.r-D*model.beta*model.lam-0.5*model.gam*model.beta**2*D**2-cw
    vola = model.beta**2/(grid.dr**2)
    corr_xr = model.beta**2*D/(2*grid.dr)
    drift_r = model.kappa*((model.theta+model.beta*model.lam/model.kappa)*np.ones(grid.Nr + 1)-grid.r)/(2*grid.dr)
    
    coe_1 = grid.dt*(-(1-model.gam)*corr_xr+drift_r+0.5*vola)
    coe_2 = 1+grid.dt*((1-model.gam)*drift_x-vola)
    coe_3 = grid.dt*((1-model.gam)*corr_xr-drift_r+0.5*vola)
    return coe_1, coe_2, coe_3


# Define function for the aggregator value 
def aggregator(cw, g, psi):
    # Case distinction for psi
    if psi !=1:
        theta_pref = (1-model.gam)/(1-1/psi)
        value = model.delta*theta_pref*(np.multiply(cw**(1-1/psi),g**(1-1/theta_pref))-g)
    else:
        value = (1-model.gam)*model.delta*np.multiply(g, np.log(cw))
    return value


# Set up grid for saving data points
psi_max = len(model.psi_list)
mult_t = 100
grid_save_data = np.arange(0, grid.Nt+1, mult_t)
Nt_save = int(grid.Nt/mult_t)
plot_t = grid.dt * grid_save_data

# Initialize value function and policies
g = np.zeros((psi_max, grid.Nr + 1, Nt_save+1))
cw = np.zeros_like(g)
D = np.zeros_like(g)

k=0
for psi in model.psi_list:
    print(f"Loop psi ={psi}")
    start_time = time.time()
    # Value function at maturity
    if psi != 1:
        theta_pref = (1-model.gam)/(1-1/psi)
        g[k, :, -1] = model.eps**((1-model.gam)/(psi-1))*model.delta**(1/theta_pref)*np.ones(grid.Nr + 1)
        g_ns_old = g[k, :, -1].copy()
    else:
        g[k, :, -1] = np.ones(grid.Nr + 1)
        g_ns_old = g[k, :, -1].copy()
        
    # Policies at maturity
    cw_ns, D_ns = policy(g_ns_old, psi)
    cw[k, :, -1] = cw_ns
    cw_ns_old = cw_ns.copy()
    D[k, :, -1] = D_ns
    D_ns_old = D_ns.copy()
    
    # Solve HJB using finite differences
    m=Nt_save-1
    for j in reversed(range(grid.Nt)):
        
        t = j * grid.dt + grid.tmin
        coe_1, coe_2, coe_3 = coefficients(cw_ns_old, D_ns_old)
        
        # Compue value function a previous time step
        g_ns = coe_2 * g_ns_old + coe_1 * np.roll(g_ns_old, -1) + coe_3 * np.roll(g_ns_old, 1)+grid.dt*aggregator(cw_ns_old, g_ns_old, psi)
        
        # Linear extrapolation of f at lower bound
        slope_lb = (g_ns[1] - g_ns[2]) / (grid.r[1] - grid.r[2])
        intercept_lb = g_ns[1] - slope_lb * grid.r[1]
        g_ns[0] = slope_lb * grid.r[0] + intercept_lb
        
        # Linear extrapolation of f at upper bound
        slope_ub = (g_ns[-2] - g_ns[-3]) / (grid.r[-2] - grid.r[-3])
        intercept_ub = g_ns[-2] - slope_ub * grid.r[-2]
        g_ns[-1] = slope_ub * grid.r[-1] + intercept_ub
        
        # Save values for value function
        g_ns_old = g_ns.copy()

        # Save values for policy
        cw_ns, D_ns = policy(g_ns, psi)

        # Set values for next iteration step
        cw_ns_old = cw_ns.copy()
        D_ns_old = D_ns.copy()
        
        # Save values
        if j in grid_save_data:
            
            g[k, :, m] = g_ns
            cw[k, :, m] = cw_ns
            D[k, :, m] = D_ns
            m=m-1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    k=k+1
print('Done Value Function Iteration')

# Plot results
colors = ['black', (0.5, 0.7, 0.2), (0.3, 0.2, 0.6), (0.8, 0.2, 0.4)]
labels_EIS = ['EIS $= 0.1$','EIS $= 0.5$', 'EIS $= 1.0$', 'EIS $= 1.5$']

plt.figure()
for i in range(4):
    plt.plot(grid.r, np.log(g[i, :, 0]), color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Function g(t, r) at $t=0$")
plt.xlabel("State $r_t$")
plt.ylabel(r"$\ln(g(0, r_t))$")
plt.legend()
plt.grid(False)
plt.xlim([-0.05, 0.3])

plt.figure()
for i in range(4):
    plt.plot(grid.r, cw[i, :, 0], color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Consumption-Wealth ratio at $t=0$")
plt.xlabel(r"State $r_t$")
plt.ylabel(r"$cw(0, r_t)$")
plt.legend()
plt.grid(False)
cwmax = cw[:,:, 0].max()
plt.xlim([-0.05, 0.3])

plt.figure()
for i in range(4):
    plt.plot(grid.r, D[i, :, 0], color=colors[i], linewidth=2, label=labels_EIS[i])
plt.title(r"Optimal Portfolio Duration at $t=0$")
plt.xlabel("State $r_t$")
plt.ylabel(r"$D(0, r_t)$")
plt.legend()
plt.grid(False)
plt.xlim([-0.05, 0.3])

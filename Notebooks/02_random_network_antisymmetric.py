# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulation of randomly connected RNN with antisymmetric iSTDP
#
# This notebook simulates a random network of condunctance-based leaky integrate-and-fire (LIF) neurons under an antysimmetric covariance-based inhibitory spike-timing dependent plasticity (iSTDP) rule. The simuation is based on Brian2 (https://brian2.readthedocs.io).
# **This simulation replicates the results in Fig 3 G-K of the main paper**
#
# When using this code, please cite our work.
#
# > Festa, Dylan, Cusseddu, Claudia and Gjorgjieva, Julijana (2024) ‘Structured stabilization in recurrent neural circuits through inhibitory synaptic plasticity’. bioRxiv, p. 2024.10.12.618014. Available at: https://doi.org/10.1101/2024.10.12.618014.
#
# This notebook is intended as a demonstration. Athough it contains the network simulation in full, it does not show the full analysis of the output data and results may differ due to random initialization. See main README for details.

# %% [markdown]
# ## Import packages
#
# (if working locally, refer to *installation_instructions.md* to set up the local environment)

# %%
# !pip install brian2
# !pip install matplotlib
# import packages
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

# %% [markdown]
# Set random seed

# %%
import random
random.seed(1)

# %% [markdown]
# ## Network and iSTDP parameters

# %%
NE = 900         # Number of excitatory cells
NI = 100          # Number of inhibitory cells
tau_ampa = 5.0 # Glutamatergic synaptic time constant (ms)
tau_gaba = 10.0 # GABAergic synaptic time constant (ms)
# simulation time 
simtime_wup = 10.0 # Warmup time
nsegments = 3
simtime_spikerecorder = 30.0
simtime_segment = 500.0
# ###########################################
# Neuron model
# ###########################################
gl = 10.0
el = -60.0
er = -80.0
vt = -50.
memc = 200.0  # Membrane capacitance
# backgound currents
bg_current_e = 0.0 # (pA)
bg_current_i = 0.0 # (pA)
# shared input for all population
n_input_shared = 50
rate_input_shared = 50.0 # (Hz)
w_insh_e = 1.0 # (nS)
w_insh_i = 0.0 # (nS)
# independent exc input
w_inindep_e = 1.0 # (nS)
n_input_indep_e = 250 # 
rate_input_indep_e = 50.0 # (Hz)
# independent inh input
w_inindep_i = 0.1 # (mV)
n_input_indep_i = 250
rate_input_indep_i = 50.0 # (Hz)
# Sparseness of synaptic connections
connection_prob_ee = 0.2
connection_prob_ei = 0.1
connection_prob_ie = 1.0
connection_prob_ii = 1.0 # denser is better
# connection weights
w_ee, w_ei, w_ie, w_ii = 1.0 , 1.0 , 1.8 , 0.3 # w_ie=3
w_max = 80               # Maximum inhibitory weight
# STDP parameters (antisymmetric, with some rate-regulation included)
A0learn = 5E-3
theta = -0.7
gamma = 0.5
alpha_pre = -0.7
alpha_post = 0.2
tauplus_stdp = 60.0    # STDP time constant (ms)

# %% [markdown]
# ## Network simulation code
#
# The block below runs the full network simulation in Brian2. Note that the iSTDP rule is defined by the equations in `eq_on_pre` and `eq_on_post`, corresponding to Eq 7 in the publication.
#
# **The simulation takes over 50 min on Google Colab, and 20 to 30 min on a personal computer**

# %%
# %%time
print(f'NE is {NE} and NI is {NI}')

eqs_neurons_e = '''
dv/dt = (-(gl*nsiemens)*(v-el*mV)-(g_ampa*v+g_gaba*(v-er*mV))+bg_current_e*pA)/(memc*pfarad) : volt (unless refractory)
dg_ampa/dt = -g_ampa/(tau_ampa*ms) : siemens
dg_gaba/dt = -g_gaba/(tau_gaba*ms) : siemens
'''

eqs_neurons_i = '''
dv/dt = (-(gl*nsiemens)*(v-el*mV)-(g_ampa*v+g_gaba*(v-er*mV))+bg_current_i*pA)/(memc*pfarad) : volt (unless refractory)
dg_ampa/dt = -g_ampa/(tau_ampa*ms) : siemens
dg_gaba/dt = -g_gaba/(tau_gaba*ms) : siemens
'''

# ###########################################
# Initialize neuron group
# ###########################################
Pe = NeuronGroup(NE, model=eqs_neurons_e, threshold='v > vt*mV',
reset='v=el*mV', refractory=5*ms, method='euler')

Pi = NeuronGroup(NI, model=eqs_neurons_i, threshold='v > vt*mV',
reset='v=el*mV', refractory=5*ms, method='euler')

# shared input
Pshared = PoissonGroup(n_input_shared, rates=rate_input_shared*Hz)
con_shared_e = Synapses(Pshared, Pe, on_pre='g_ampa += w_insh_e*nS')
con_shared_e.connect()
con_shared_i = Synapses(Pshared, Pi, on_pre='g_ampa += w_insh_i*nS')
con_shared_i.connect()

# independent input
PIe = PoissonInput(Pe, 'g_ampa', n_input_indep_e, rate_input_indep_e*Hz, weight=w_inindep_e*nS)
PIi = PoissonInput(Pi, 'g_ampa', n_input_indep_i, rate_input_indep_i*Hz, weight=w_inindep_i*nS)

# ##########################################
# Connecting the network
# ###########################################
con_ee = Synapses(Pe, Pe, on_pre='g_ampa += w_ee*nS')
con_ee.connect(condition='i!=j', p=connection_prob_ee)
con_ei = Synapses(Pe, Pi, on_pre='g_ampa += w_ei*nS')
con_ei.connect(p=connection_prob_ei)
con_ii = Synapses(Pi, Pi, on_pre='g_gaba += w_ii*nS')
con_ii.connect(condition='i!=j', p=connection_prob_ii)

# ###########################################
# Inhibitory Plasticity
# ###########################################
A0 = 0.0 # start with no learning

# derived parameters
tauminus_stdp = gamma*tauplus_stdp
# NOT scaled by A0 here (since it controls learning on/off)
Aplus = float(1/tauplus_stdp)*1E3
Aminus = float(theta/tauminus_stdp)*1E3

# simple traces for pre- and postsynaptic activity
# (that need to be rescaled)

eqs_stdp_inhib = '''
    w : 1
    dtrace_pre_plus/dt=-trace_pre_plus/(tauplus_stdp*ms) : 1 (event-driven)
    dtrace_post_minus/dt=-trace_post_minus/(tauminus_stdp*ms) : 1 (event-driven)
    '''
eq_on_pre = '''
    trace_pre_plus += 1.0
    w = clip(w + A0*(alpha_pre + Aminus*trace_post_minus), 0, w_max)
    g_gaba += w*nS
    '''
eq_on_post = '''
    trace_post_minus += 1.0
    w = clip(w + A0*(alpha_post + Aplus*trace_pre_plus), 0, w_max)
    '''

con_ie = Synapses(Pi, Pe, model=eqs_stdp_inhib,on_pre=eq_on_pre,on_post=eq_on_post)
con_ie.connect()
con_ie.w = w_ie

# ###########################################
# Setting up monitors
# ###########################################
wei_mon = StateMonitor(con_ie, 'w', record=True, dt=5.0*second)
wei_mon.active = False
pop_re_mon = PopulationRateMonitor(Pe)
pop_re_mon.active = False
pop_ri_mon = PopulationRateMonitor(Pi)
pop_ri_mon.active = False

sme = SpikeMonitor(Pe)
smi = SpikeMonitor(Pi)
sme.active = False
smi.active = False

# ###########################################
# Warmup: no plasticity, low noise for all
# ###########################################
print('Running warmup')
run(simtime_wup*second, report='text')

# ###########################################
# Record spikes right after warmup, plasticity still off
# ###########################################
print('Record after warmup, plasticity off')
sme.active = True
smi.active = True
pop_re_mon.active = True
pop_ri_mon.active = True
wei_mon.active = True 
run(simtime_spikerecorder*second, report='text')

# ###########################################
# For loop on segments
# ###########################################
# plasticity on
A0 = A0learn
for thesegment in range(nsegments):
    print('Now running segment ', thesegment + 1, ' of ', nsegments, '\n')
    # recorder off
    sme.active = False
    smi.active = False
    run(simtime_segment*second, report='text')
    # recorder on
    sme.active = True
    smi.active = True
    run(simtime_spikerecorder*second, report='text')

print('******* \n All runs completed!\n*******')

# %% [markdown]
# ## Results
#
#
# ### Population rates

# %%
# Extract population rates and time from monitors
pop_re_times = pop_re_mon.t / second
pop_re_rates = pop_re_mon.smooth_rate(window='flat',width=0.5*second) / Hz

pop_ri_times = pop_ri_mon.t / second
pop_ri_rates = pop_ri_mon.smooth_rate(window='flat',width=0.5*second) / Hz

# Create the plot
nplot = 200
idxplot = np.linspace(start=1,stop=len(pop_re_times)-1,num=nplot).round().astype(int)
plt.figure(figsize=(10, 5))
plt.plot(pop_re_times[idxplot]/60, pop_re_rates[idxplot], label='Excitatory',color='blue')
plt.plot(pop_ri_times[idxplot]/60, pop_ri_rates[idxplot], label='Inhibitory',color='red')
plt.xlabel('time (min)')
plt.ylabel('population Rate (Hz)')
plt.title('Population Rates over Time')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# This plot shows that, before plasticity starts, population rates are near saturation, therefore the system is unstable. Inhibitory plasticity reduces the firing rates to much lower levels. The circuit is therefore inhibition-stabilized. The very sharp drop in rates here is due to the parameter choice in the inhibitory plasticity rule.

# %% [markdown]
# ### Distribution of final mutual vs unidirectional weights
#
# (this needs some improvement in the code, but the results is as intended)

# %%
# Initialize empty lists for mutual and unidirectional weights
w_ie_mutual = []
w_ie_unidirectional = []
# count number of connections too
n_mutual = 0
n_uni = 0
# build exc to inh weight matrix in full
w_ei_dense = np.zeros((NE,NI))
w_ei_dense[con_ei.i,con_ei.j]=w_ei

# now iterate over inh to exc connections
n_con_ie = len(con_ie.w)
for idx_ie_i, idx_ie_j,k in zip(con_ie.i, con_ie.j,range(n_con_ie)):
    # Check if a reciprocal connection exists in con_ei
    is_mutual =  w_ei_dense[idx_ie_j,idx_ie_i] > 1E-4
    this_w = con_ie.w[k]        
    if is_mutual:
        n_mutual+=1
        w_ie_mutual.append(this_w)
    else:
        n_uni+=1
        w_ie_unidirectional.append(this_w)

# print total mutual and total unidirectional
print(f"found {n_mutual} mutual connections and {n_uni} unidirectional connections. In total {n_mutual+n_uni}")
        
# Convert lists to numpy arrays
w_ie_mutual = np.array(w_ie_mutual)
w_ie_unidirectional = np.array(w_ie_unidirectional)
# Calculate histogram values and normalize
hist_mutual, bins_mutual = np.histogram(w_ie_mutual, bins=np.arange(0,25, 1), density=True)
hist_unidirectional, bins_unidirectional = np.histogram(w_ie_unidirectional, bins=np.arange(0, 25, 1), density=True)

# Replace zero values with a small positive value to avoid log(0) error
hist_mutual[hist_mutual == 0] = 1e-6 
hist_unidirectional[hist_unidirectional == 0] = 1e-6

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Normal scale plot
ax1.hist(bins_mutual[:-1], bins_mutual, weights=hist_mutual, color='brown', alpha=0.7, label='Mutual')
ax1.hist(bins_unidirectional[:-1], bins_unidirectional, weights=hist_unidirectional, color='darkcyan', alpha=0.7, label='Unidirectional')
ax1.set_xlabel('Synaptic Weight')
ax1.set_ylabel('Normalized Frequency')
ax1.set_title('Distribution of Weights (Normal Scale)')
ax1.legend()

# Logarithmic scale plot
ax2.hist(bins_mutual[:-1], bins_mutual, weights=hist_mutual, color='brown', alpha=0.7, label='Mutual')
ax2.hist(bins_unidirectional[:-1], bins_unidirectional, weights=hist_unidirectional, color='darkcyan', alpha=0.7, label='Unidirectional')
ax2.set_yscale('log')  # Set logarithmic y-axis
ax2.set_xlabel('Synaptic Weight')
ax2.set_ylabel('density (log scale)')
ax2.set_title('Distribution of Weights (Logarithmic Scale)')
ax2.legend()

# Display plot
plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# %% [markdown]
# When examining the inh to exc weights after learning, one can clearly see a net separation between mutual connections and unidirectional connections. The choice of antisymmetric, covariance-dominated inhibitory plasticity results in much stronger unidirectional connections than mutual connections.

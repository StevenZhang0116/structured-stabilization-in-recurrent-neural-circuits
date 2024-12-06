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
# # Simulation of ring network with two interneuron subclasses
#
# This notebook simulates a recurrent network of condunctance-based leaky integrate-and-fire (LIF) neurons with a non-random, ring-like connectivity. Inhibitory neurons are divided in two subclasses, with either a symmetric or neurons under an antysimmetric covariance-based inhibitory spike-timing dependent plasticity (iSTDP) rule. The simuation is based on Brian2 (https://brian2.readthedocs.io).
# **This simulation replicates the results in Fig 4 A-D of the main paper**
#
# Running the simulation takes about 25 min on a normal compure, but it might take up to 40 minuts on Google Colab. The simulation results are saved on a binary file.
#
# When using this code, please cite our work.
#
# > Festa, Dylan, Cusseddu, Claudia and Gjorgjieva, Julijana (2024) ‘Structured stabilization in recurrent neural circuits through inhibitory synaptic plasticity’. bioRxiv, p. 2024.10.12.618014. Available at: https://doi.org/10.1101/2024.10.12.618014.
#
# This notebook is intended as a demonstration. Athough it contains the full network training, it does not show the full analysis of the output data and results may differ due to random initialization. It also omits the surround-suppression example. See main README for details.

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
import os
import pickle

# %% [markdown]
# Set random seed

# %%
import random
random.seed(1)


# %% [markdown]
# ## Auxiliary functions that define the ring connectivity structure

# %%
def vonmisesnotnorm(theta, mu, k):
    """
    Calculate the value of the un-normalized von Mises distribution at a given angle.
    Always peaks at 1.0

    Parameters:
    theta (float): The angle at which to evaluate the distribution.
    mu (float): The mean angle of the distribution.
    k (float): The concentration parameter of the distribution.

    Returns:
    float: The value of the normalized von Mises distribution at the given angle.
    """
    assert 0.0 <= theta <= 2 * np.pi, "θ must be in the range [0, 2π]"
    return np.exp(k * np.cos(theta - mu)) / np.exp(k)

def vonmisesnorm(theta, mu, k):
    """
    Calculate the value of the von Mises distribution at a given angle.

    Parameters:
    theta (float): The angle at which to evaluate the distribution.
    mu (float): The mean angle of the distribution.
    k (float): The concentration parameter of the distribution.

    Returns:
    float: The value of the normalized von Mises distribution at the given angle.
    """
    assert 0.0 <= theta <= 2 * np.pi, "θ must be in the range [0, 2π]"
    return np.exp(k * np.cos(theta - mu)) / (2 * np.pi * i0(k))

def assign_angles(n):
    """
    Generates equally spaced angles.

    Parameters:
        n (int): The number of angles to generate.

    Returns:
        numpy.ndarray: An array of equally spaced angles.
    """
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]  
    return angles


def make_ring_pre_post_fixed_sum(npre, npost, wsum, k_vm, w_thresh=0.0,avoid_self_connections=False):    
    """
    Creates connectivity between pre and post layers in a ring model with a fixed sum of weights along the rows

    Parameters:
        npre (int): Number of pre neurons
        npost (int): Number of post neurons
        wsum (float): Sum of weights.
        k_vm (float): Concentration parameter for the von Mises distribution.
        w_thresh (float, optional): Threshold value for weights. Defaults to 0.0.
        avoid_self_connections (bool, optional): Avoid self connections. Defaults to False.

    Returns:
        numpy.ndarray: Connectivity matrix between pre and post layers.

    """
    muspre = assign_angles(npre)
    muspost = assign_angles(npost)
    w = np.zeros((npost, npre))
    for i in range(npost):
        for j in range(npre):
            if avoid_self_connections and i == j:
                continue
            else:
                w[i, j] = vonmisesnorm(np.abs(muspre[j] - muspost[i]), 0.0, k_vm)
               
    w[w < w_thresh] = 0.0
   
    expected_sum = 1 / (np.pi*2/npost)
    if avoid_self_connections:
        expected_sum -= vonmisesnorm(0, 0, k_vm) # Subtract self-connection if present
    w *= wsum/expected_sum
    return w

  
def wmat_to_pre_post_list(wmat, w_threshold=1E-6):
    """Converts weight matrix to list representation"""
    post_all, pre_all = np.where(wmat > w_threshold)  # Efficiently find non-zero indices
    w_all = wmat[post_all,pre_all]
    return pre_all.tolist(), post_all.tolist(), w_all.tolist() 

def make_ring_pre_post_fixedsum_list(npre, npost, wsum, k_vm, w_thresh=0.0,avoid_self_connections=False):
    """
    Generates a weight matrix for a ring model with fixed sum of weights and converts it into a pre-post list.

    Args:
        npre (int): Number of presynaptic neurons.
        npost (int): Number of postsynaptic neurons.
        wsum (float): Sum of weights.
        k_vm (float): Scaling factor for Von Mises
        w_thresh (float, optional): Threshold weight value. Defaults to 0.0.

    Returns:
        list: Pre-post list representation of the weight matrix.
             format is (all_pre_indexes,all_post_indexes,all_weights)      

    """
    wmat = make_ring_pre_post_fixed_sum(npre, npost, wsum, k_vm, w_thresh,avoid_self_connections)
    return wmat_to_pre_post_list(wmat, w_thresh)


# %% [markdown]
# ## Network and iSTDP parameters

# %%
NE = 800         # Number of excitatory cells
NP = 100         # number of PV cells (symmetric iSTDP)
NS = 100         # number of SST cells (antisymmetric iSTDP)


# simulation time
time_warmup = 3.0 # Warmup time (seconds)
time_training = 1500.0 # Simulation time (seconds)
time_checkspikes = 60.0 # record spikes of exc population here

# ###########################################
# Neuron model
# ###########################################
tau_ampa = 5.0 # Glutamatergic synaptic time constant (ms)
tau_gaba = 10.0 # GABAergic synaptic time constant (ms)
gl = 10.0
el = -60.0
er = -80.0
vt = -50.0
memc = 200.0  # Membrane capacitance
# backgound currents
bg_current_e = 1000.0 # External current (pA)
bg_current_p = 0.0
bg_current_s = 0.0
# noisy input for exc
n_input_e = 200
rate_input_e = 80 # (Hz)
strength_input_e = 1.0 # (mV)
# noisy input for PV
n_input_p = 200
rate_input_p = 40 # (Hz)
strength_input_p = 1.0 # (mV)
# noisy input for SST
n_input_s = 200
rate_input_s = 40 # (Hz)
strength_input_s = 1.0 # (mV)
    
# connection sparseness (inh. only)
connprob_sp = 0.9
connprob_ss = 0.9
connprob_ps = 0.9
connprob_pp = 0.9
# ring parameters (lower is broader)
k_ring_ee=10.0 
k_ring_ep=5.0 
k_ring_es=10.0 
# connection weights
w_ee_sum = 145.0 # exc to exc, SUM
w_ep_sum,w_es_sum = 145.0,145.0 # exc to inh SUM
w_ps,w_sp =1.0,1.0
w_ss,w_pp = 2.5,2.5
w_pe,w_se= 3.0,3.0 # the plastic ones
# Maximum inhibitory weight
weigh_max = 80

# STDP parameters for PV to exc
A0learn_p = 2E-3
theta_p = -1.0
gamma_p = 20.0
alpha_pre_p = -0.2
alpha_post_p = 1.0
tauplus_stdp_p = 30.0    # STDP time constant (ms)

# STDP parameters for SST to exc
A0learn_s = 1E-3
theta_s = -1.0
gamma_s = 1.0
alpha_pre_s = -0.2
alpha_post_s = 1.0
tauplus_stdp_s = 30.0    # STDP time constant (ms)

# Simulation details
save_id = 'ring_structure_two_interneurons'

# %% [markdown]
# ## Network simulation code
#
# The block below runs the full network simulation in Brian2. Note that the iSTDP rule is defined by the equations in `eq_on_pre` and `eq_on_post`, corresponding to Eq 7 in the publication.
#
# **The simulation takes over 50 min on Google Colab, and 20 to 30 min on a personal computer**

# %%
print(f'NE is {NE}, NP is {NP}, NS is {NS}')

eqs_neurons_e='''
input_contrast : 1
dv/dt=(-(gl*nsiemens)*(v-el*mV)-(g_ampa*v+g_gaba*(v-er*mV))+bg_current_e*pA)/(memc*pfarad) : volt (unless refractory)
dg_ampa/dt = -g_ampa/(tau_ampa*ms) : siemens
dg_gaba/dt = -g_gaba/(tau_gaba*ms) : siemens
'''

eqs_neurons_p='''
dv/dt=(-(gl*nsiemens)*(v-(el*mV))-(g_ampa*v+g_gaba*(v-(er*mV)))+bg_current_p*pA)/(memc*pfarad) : volt (unless refractory)
dg_ampa/dt = -g_ampa/(tau_ampa*ms) : siemens
dg_gaba/dt = -g_gaba/(tau_gaba*ms) : siemens
'''

eqs_neurons_s='''
dv/dt=(-(gl*nsiemens)*(v-(el*mV))-(g_ampa*v+g_gaba*(v-(er*mV)))+bg_current_s*pA)/(memc*pfarad) : volt (unless refractory)
dg_ampa/dt = -g_ampa/(tau_ampa*ms) : siemens
dg_gaba/dt = -g_gaba/(tau_gaba*ms) : siemens
'''

# ###########################################
# Initialize neuron group
# ###########################################
Pe = NeuronGroup(NE, model=eqs_neurons_e, threshold='v > (vt*mV)',
                 reset='v=(el*mV)', refractory=5*ms, method='euler')
Pp = NeuronGroup(NP, model=eqs_neurons_p, threshold='v > (vt*mV)',
                 reset='v=(el*mV)', refractory=5*ms, method='euler')
Ps = NeuronGroup(NS, model=eqs_neurons_s, threshold='v > (vt*mV)',
                 reset='v=(el*mV)', refractory=5*ms, method='euler')

PIe = PoissonInput(Pe, 'g_ampa', n_input_e, rate_input_e*Hz, weight=strength_input_e*nS)
PIp = PoissonInput(Pp, 'g_ampa', n_input_p, rate_input_p*Hz, weight=strength_input_p*nS)
PIs = PoissonInput(Ps, 'g_ampa', n_input_s, rate_input_s*Hz, weight=strength_input_s*nS)

# ##########################################
# Connecting the network
# ###########################################

# exc to exc and to inh
con_ee = Synapses(Pe, Pe,model='w:1', on_pre='g_ampa += w*nS') 
con_ep = Synapses(Pe, Pp,model='w:1', on_pre='g_ampa += w*nS')
con_es = Synapses(Pe, Ps,model='w:1', on_pre='g_ampa += w*nS')
# inh to inh 
con_ps = Synapses(Pp, Ps, on_pre='g_gaba += w_ps*nS')
con_sp = Synapses(Ps, Pp, on_pre='g_gaba += w_sp*nS')
con_pp = Synapses(Pp, Pp, on_pre='g_gaba += w_pp*nS')
con_ss = Synapses(Ps, Ps, on_pre='g_gaba += w_ss*nS')


# ###########################################
# Inhibitory to inhibitory connections
# ###########################################
con_sp.connect(p=connprob_sp)
con_ps.connect(p=connprob_ps)
con_pp.connect(p=connprob_pp)
con_ss.connect(p=connprob_ss)


# ###################################################
# Excitatory connections implement the ring structure
# ##################################################

# exc to exc
idxee_pre,idxee_post,wvals_ee = make_ring_pre_post_fixedsum_list(
    NE,NE,w_ee_sum,k_ring_ee,w_thresh=0.001,avoid_self_connections=True)
con_ee.connect(i=idxee_pre,j=idxee_post)
con_ee.w = wvals_ee

# exc to pv
idxep_pre,idxep_post,wvals_ep = make_ring_pre_post_fixedsum_list(
    NE,NP,w_ep_sum,k_ring_ep,w_thresh=0.001,avoid_self_connections=False)
con_ep.connect(i=idxep_pre,j=idxep_post)
con_ep.w = wvals_ep

# exc to sst
idxes_pre,idxes_post,wvals_es = make_ring_pre_post_fixedsum_list(
    NE,NS,w_es_sum,k_ring_es,w_thresh=0.001,avoid_self_connections=False)
con_es.connect(i=idxes_pre,j=idxes_post)
con_es.w = wvals_es


# ###########################################
# Inhibitory Plasticity
# ###########################################
A0_p = 0.0 # start with no learning
A0_s = 0.0 # start with no learning

# derived parameters
tauminus_stdp_p = gamma_p*tauplus_stdp_p
tauminus_stdp_s = gamma_s*tauplus_stdp_s
# NOT scaled by A0 here (since it controls learning on/off)
Aplus_p = float(1/tauplus_stdp_p)*1E3 # time was in ms, needs to be seconds
Aminus_p = float(theta_p/tauminus_stdp_p)*1E3
Aplus_s = float(1/tauplus_stdp_s)*1E3
Aminus_s = float(theta_s/tauminus_stdp_s) *1E3


# simple traces for pre- and postsynaptic activity
# (that need to be rescaled)

eqs_stdp_inhib_p = '''
w : 1
dtrace_pre_plus/dt=-trace_pre_plus/(tauplus_stdp_p*ms) : 1 (event-driven)
dtrace_pre_minus/dt=-trace_pre_minus/(tauminus_stdp_p*ms) : 1 (event-driven)
dtrace_post_plus/dt=-trace_post_plus/(tauplus_stdp_p*ms) : 1 (event-driven)
dtrace_post_minus/dt=-trace_post_minus/(tauminus_stdp_p*ms) : 1 (event-driven)
'''

con_pe = Synapses(Pp, Pe, model=eqs_stdp_inhib_p,
    on_pre='''
        trace_pre_plus += 1.0
        trace_pre_minus += 1.0
        w = clip(w + A0_p*(alpha_pre_p + Aplus_p*trace_post_plus + Aminus_p*trace_post_minus), 0, weigh_max)
        g_gaba += w*nS''',
    on_post='''
        trace_post_plus += 1.0
        trace_post_minus += 1.0
        w = clip(w + A0_p*(alpha_post_p + Aplus_p*trace_pre_plus + Aminus_p*trace_pre_minus), 0, weigh_max)''')
con_pe.connect()
con_pe.w = w_pe

    
eqs_stdp_inhib_s = '''
w : 1
dtrace_plus/dt=-trace_plus/(tauplus_stdp_s*ms) : 1 (event-driven)
dtrace_minus/dt=-trace_minus/(tauminus_stdp_s*ms) : 1 (event-driven)
'''

con_se = Synapses(Ps, Pe, model=eqs_stdp_inhib_s,
    on_pre='''
        trace_plus += 1.0
        w = clip(w + A0_s*(alpha_pre_s + Aminus_s*trace_minus), 0, weigh_max)
        g_gaba += w*nS''',
    on_post='''
        trace_minus += 1.0
        w = clip(w + A0_s*(alpha_post_s + Aplus_s*trace_plus), 0, weigh_max)''')
con_se.connect()
con_se.w = w_se

# ###########################################
# Setting up monitors
# ###########################################

# population rates
pop_re_mon = PopulationRateMonitor(Pe)
pop_rp_mon = PopulationRateMonitor(Pp)
pop_rs_mon = PopulationRateMonitor(Ps)
    
# exc spikes
spk_e_mon = SpikeMonitor(Pe)
# pv spikes
spk_p_mon = SpikeMonitor(Pp)
# sst spikes
spk_s_mon = SpikeMonitor(Ps)
    
spk_e_mon.active=False
spk_p_mon.active=False
spk_s_mon.active=False

pop_re_mon.active=False
pop_rp_mon.active=False
pop_rs_mon.active=False

# %% [markdown]
# ## Running the network

# %%
# %%time
# ###########################################
# Run warmup (no plasticity, no recording)
# ###########################################print('Running warmup')
print('Running warmup')
run(time_warmup*second)
print('Warmup done, now running with plasticity')

# ###########################################
# Run with plasticity and monitors, except spiketimes
# ###########################################
A0_p = A0learn_p
A0_s = A0learn_s

pop_re_mon.active=True
pop_rp_mon.active=True
pop_rs_mon.active=True

 
run(time_training*second, report='text',report_period=60.0*second)

print('weight training complete, now recording spike trains...')

# ###########################################
# Now final run with spiketime recorder
# ###########################################
spk_e_mon.active=True
spk_p_mon.active=True
spk_s_mon.active=True
run(time_checkspikes*second, report='text')

print('******* \nRuns completed, preparing objects to save\n*******')

def fill_like(vec,val):
    n= len(vec)
    return np.full(n,val)
        
    
w_e2e_components = {
    'NE' : NE,
    'con_ee_ijw': np.stack(
            (np.array(con_ee.i),np.array(con_ee.j),np.array(con_ee.w)),axis=-1),
}
w_e2p_components = {
    'NE' : NE,
    'NP':NP,
    'con_ep_ijw': np.stack(
        (np.array(con_ep.i),np.array(con_ep.j),np.array(con_ep.w)),axis=-1),
}
w_p2e_components = {
    'NP':NP,
    'NE':NE,
    'con_pe_ijw': np.stack(
        (np.array(con_pe.i),np.array(con_pe.j),con_pe.w),axis=-1),
    }
w_p2p_components = {
    'NP':NP,
    'con_pp_ijw': np.stack(
            (np.array(con_pp.i),np.array(con_pp.j),fill_like(con_pp.i,w_pp)),axis=-1),
}

w_e2s_components = {
    'NE' : NE,
    'NS':NS,
    'con_es_ijw': np.stack(
        (np.array(con_es.i),np.array(con_es.j),np.array(con_es.w)),axis=-1),
    }
w_s2e_components = {
    'NS':NS,
    'NE':NE,
    'con_se_ijw': np.stack(
        (np.array(con_se.i),np.array(con_se.j),con_se.w),axis=-1),
    }
w_s2s_components = {
    'NS':NS,
    'con_ss_ijw': np.stack(
        (np.array(con_ss.i),np.array(con_ss.j),fill_like(con_ss.i,w_ss)),axis=-1),
    }
w_p2s_components = {
    'NP':NP,
    'NS':NS,
    'con_ps_ijw': np.stack(
        (np.array(con_ps.i),np.array(con_ps.j),fill_like(con_ps.i,w_ps)),axis=-1),
}
w_s2p_components = {
    'NS':NS,
    'NP':NP,
    'con_sp_ijw': np.stack(
        (np.array(con_sp.i),np.array(con_sp.j),fill_like(con_sp.i,w_sp)),axis=-1),
}
    
# Export population rates
pop_re_times = pop_re_mon.t / second
pop_re_rates = pop_re_mon.smooth_rate(window='gaussian',width=0.05*second) / Hz
pop_rp_times = pop_rp_mon.t / second
pop_rp_rates = pop_rp_mon.smooth_rate(window='gaussian',width=0.05*second) / Hz
pop_rs_times = pop_rs_mon.t / second
pop_rs_rates = pop_rs_mon.smooth_rate(window='gaussian',width=0.05*second) / Hz
    
print('Now saving data on file')
# %% output is a dictionary
save_dict = { 
        'ne':NE,
        'np':NP,
        'ns':NS,
        'w_e2e_components':w_e2e_components,
        'w_e2p_components':w_e2p_components,
        'w_e2s_components':w_e2s_components,
        'w_p2e_components':w_p2e_components,
        'w_s2e_components':w_s2e_components,
        'w_p2s_components':w_p2s_components,
        'w_s2p_components':w_s2p_components,
        'w_p2p_components':w_p2p_components,
        'w_s2s_components':w_s2s_components,
        'pop_re_times':pop_re_times,
        'pop_re_rates':pop_re_rates,
        'pop_rp_times':pop_rp_times,
        'pop_rp_rates':pop_rp_rates,
        'pop_rs_times':pop_rs_times,
        'pop_rs_rates':pop_rs_rates,
        'spikeindices_e':np.array(spk_e_mon.i),
        'spiketimes_e':np.array(spk_e_mon.t/second),
        'spikeindices_p':np.array(spk_p_mon.i),
        'spiketimes_p':np.array(spk_p_mon.t/second),
        'spikeindices_s':np.array(spk_s_mon.i),
        'spiketimes_s':np.array(spk_s_mon.t/second),
        }

savetemppath = f'brian2network_{save_id}.pkl'
# if file exists, create new file with random string added
while os.path.exists(savetemppath):
    savetemppath =  f'brian2network_{save_id}_{np.random.randint(1000)}.pkl'
with open(savetemppath,'wb') as f:
    pickle.dump(save_dict,f)
print(f'Created temporary file {savetemppath} to store data')

print('******* \n Run completed!\n*******')


# %% [markdown]
# ## Results
#
#
# ### Population rates

# %%
color_e='blue'
color_p='red'
color_s='orange'
nplot = 500
time_end = 25*60 # stop the plot at 25 min
idx_end = np.argmax(pop_re_times>time_end)
idxplot = np.linspace(start=1,stop=idx_end,num=nplot).round().astype(int) # subsample, as data points are too many
plt.figure(figsize=(10, 5))
plt.plot(pop_re_times[idxplot]/60, pop_re_rates[idxplot], label='Exc',color=color_e)
plt.plot(pop_rp_times[idxplot]/60, pop_rp_rates[idxplot], label='PV',color=color_p)
plt.plot(pop_rs_times[idxplot]/60, pop_rs_rates[idxplot], label='SST',color=color_s)
plt.xlabel('time (min)')
plt.ylabel('population rate (Hz)')
plt.title('population rates over time')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# This plot shows that excitatory rates are regulated by inhibitory plasticity.

# %% [markdown]
# ### Normalized connectivity profile
#
#

# %% [markdown]
# #### Utility functions to average by angular tuning

# %%
def unband_and_center(mat, indexes=None):
    nrow, ncol = mat.shape
    if indexes is None:  # only works on square matrices
        indexes = np.arange(ncol)
    assert len(indexes) == nrow, "indexes must have same length as rows of mat"
    assert np.all((0 <= indexes) & (indexes <= ncol-1)), "indexes must be in the range 0:ncol-1"
    inddiff = np.diff(indexes)
    if np.any(inddiff > 1):
        raise ValueError("indexes are not contiguous")
    ret = np.empty((nrow, ncol), dtype=mat.dtype)
    ncenter = ncol // 2 - 1
    for i in range(nrow):
        for j in range(ncol):
            js = (j + indexes[i] ) % ncol
            ret[i, (j + ncenter) % ncol] = mat[i, js]
    return ret

def mean_and_norm(mat):
    ret = np.mean(mat, axis=0)
    ret /= np.max(ret)
    return ret

def mean_cols_nodiagonal(mat):
    nrows, ncols = mat.shape
    ret = np.empty(ncols, dtype=mat.dtype)
    for j in range(ncols):
        ret[j] = np.sum(np.delete(mat[:, j], j))
    ret /= (nrows - 1)
    return ret

def unband_mean_normalize(mat, indexes=None):
    if indexes is None:
        indexes = np.arange(1, mat.shape[1] + 1)
    mat_unband = unband_and_center(mat, indexes)
    means = mean_cols_nodiagonal(mat_unband)
    return means / np.max(means)


# %% [markdown]
# #### Convert i,j,w into 2D matrices

# %%
# %%time
w_ee_full = np.zeros((NE,NE),dtype=float)
for (i,j,w) in zip(con_ee.i,con_ee.j,con_ee.w):
    w_ee_full[j,i] = w
w_es_full = np.zeros((NS,NE),dtype=float)
for (i,j,w) in zip(con_es.i,con_es.j,con_es.w):
    w_es_full[j,i] = w
w_se_full = np.zeros((NE,NS),dtype=float)
for (i,j,w) in zip(con_se.i,con_se.j,con_se.w):
    w_se_full[j,i] = w
w_ep_full = np.zeros((NP,NE),dtype=float)
for (i,j,w) in zip(con_ep.i,con_ep.j,con_ep.w):
    w_ep_full[j,i] = w
w_pe_full = np.zeros((NE,NP),dtype=float)
for (i,j,w) in zip(con_pe.i,con_pe.j,con_pe.w):
    w_pe_full[j,i] = w
    
w_ps_full = np.zeros((NS,NP),dtype=float)
for (i,j) in zip(con_ps.i,con_ps.j):
    w_ps_full[j,i] = w_ps
w_sp_full = np.zeros((NP,NS),dtype=float)
for (i,j) in zip(con_sp.i,con_sp.j):
    w_sp_full[j,i] = w_sp
w_pp_full = np.zeros((NP,NP),dtype=float)
for (i,j) in zip(con_pp.i,con_pp.j):
    w_pp_full[j,i] = w_pp
w_ss_full = np.zeros((NS,NS),dtype=float)
for (i,j) in zip(con_ss.i,con_ss.j):
    w_ss_full[j,i] = w_ss

# %% [markdown]
# #### Plot of connectivity profiles

# %%
# Shift indexes PV and SST so that they are related to the exc population
shift_idx_sst = np.array(np.linspace(0,NS-1,NE),dtype=int)
shift_idx_pv = np.array(np.linspace(0,NP-1,NE),dtype=int)

tuning_exc_all = np.linspace(-1.0,1.0,NE+1)[0:NE] # because -1 and +1 are the same point
tuning_pv_all = np.linspace(-1.0,1.0,NP+1)[0:NP]
tuning_sst_all = np.linspace(-1.0,1.0,NS+1)[0:NS]

profile_ee=mean_and_norm(unband_and_center(w_ee_full))
profile_pe=mean_and_norm(unband_and_center(w_pe_full,shift_idx_pv))
profile_ep=mean_and_norm(unband_and_center(w_ep_full.transpose(),shift_idx_pv))
profile_se=mean_and_norm(unband_and_center(w_se_full,shift_idx_sst))
profile_es=mean_and_norm(unband_and_center(w_es_full.transpose(),shift_idx_sst))

# Let's plot
color_e_to_e='#0000ff'
color_e_to_pv='#04ff00'
color_e_to_sst='#0d00a3'
color_p='red'
color_s='orange'

plt.figure(figsize=(10, 5))
plt.plot(tuning_exc_all,profile_ee, label='exc to exc',color=color_e_to_e)
plt.plot(tuning_pv_all,profile_ep, label='exc to PV',color=color_e_to_pv)
plt.plot(tuning_pv_all,profile_es, label='exc to SST',color=color_e_to_sst,linestyle='dashed',linewidth=3)
plt.plot(tuning_pv_all,-profile_pe, label='PV to exc',color=color_p)
plt.plot(tuning_sst_all,-profile_se, label='SST to exc',color=color_s)
plt.xlabel('tuning')
plt.ylabel('normalized weight (arb.u.)')
plt.title('connectivity profiles (normalized to +1,-1)')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# As expected from the results on the random network, the PV population (symmetric iSTDP) appears co-tuned with excitation, forming mutual connections. The SST neurons project their connections laterally, instead.

# %% [markdown]
# ### Shape of effective exc to exc interaction

# %% [markdown]
# The effective interaction between excitatory neurons is found by considering the expansion series of the full weight matrix.

# %%
NTOT=NE+NP+NS
idx_e = np.arange(NE)
idx_p = NE+np.arange(NP)
idx_s = NE+NP+np.arange(NS)

w_full = np.zeros((NTOT,NTOT),dtype=float)
w_full[np.ix_(idx_e,idx_e)]=w_ee_full
w_full[np.ix_(idx_p,idx_e)]=w_ep_full
w_full[np.ix_(idx_e,idx_p)]=-w_pe_full
w_full[np.ix_(idx_p,idx_p)]=-w_pp_full

w_full[np.ix_(idx_s,idx_e)]=w_es_full
w_full[np.ix_(idx_e,idx_s)]=-w_se_full
w_full[np.ix_(idx_s,idx_s)]=-w_ss_full

w_full[np.ix_(idx_s,idx_p)]=-w_ps_full
w_full[np.ix_(idx_p,idx_s)]=-w_sp_full

# Just the sum of powers of the weight matrix
w_effective = (np.eye(NTOT) 
              + w_full 
               + w_full @ w_full 
               + w_full @ w_full @ w_full 
               + w_full @ w_full @ w_full @ w_full
               + w_full @ w_full @ w_full @ w_full  @ w_full  
               + w_full @ w_full @ w_full @ w_full  @ w_full @ w_full)
# selec only exc to exc portion
w_effective_ee = w_effective[np.ix_(idx_e,idx_e)]
# now I take the average, as I did with the weight matrix for the previous figure 
profile_ee_effective=mean_and_norm(unband_and_center(w_effective_ee))
# z-score the result
profile_ee_effective_zscored = (profile_ee_effective - profile_ee_effective.mean()) / profile_ee_effective.std()

plt.figure(figsize=(10, 5))
plt.plot(tuning_exc_all,profile_ee_effective_zscored,color='black',linewidth=4)
plt.xlabel('tuning')
plt.ylabel('effective interaction exc to exc (z-scored)')
plt.title('profile of average excitatory to excitatory interactions')
plt.grid(True)
plt.show()

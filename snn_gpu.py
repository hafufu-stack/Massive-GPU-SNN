"""
Massive Spiking Neural Network Simulation using Numba & CUDA
------------------------------------------------------------
Author: [Your Name / Handle]
Description: 
  Simulates millions of Izhikevich neurons on a GPU using Numba.
  Includes a simplified STDP-like plasticity mechanism.
  Optimized for NVIDIA RTX 30/40/50 series.
"""

from numba import cuda
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import math

# --- Configuration ---
N_AGENTS = 1000  # Number of agents (if applicable)
NEURONS_PER_AGENT = 5000
N_NEURONS = N_AGENTS * NEURONS_PER_AGENT  # Total Neurons (e.g., 5,000,000)
SYNAPSES_PER_NEURON = 50 # Synapses per neuron
TOTAL_SYNAPSES = N_NEURONS * SYNAPSES_PER_NEURON

EPOCHS = 100
STEPS_PER_EPOCH = 100
OUTPUT_DIR = "simulation_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"=== Massive SNN Simulation ===")
print(f"Total Neurons:  {N_NEURONS:,}")
print(f"Total Synapses: {TOTAL_SYNAPSES:,}")

# --- Initialize Host Memory ---
print("Initializing host memory...")
# Izhikevich variables
v_host = np.ones(N_NEURONS, dtype=np.float32) * -65.0
u_host = np.zeros(N_NEURONS, dtype=np.float32)

# Neuron Parameters (Randomized mix of types)
a_host = np.ones(N_NEURONS, dtype=np.float32) * 0.02
b_host = np.ones(N_NEURONS, dtype=np.float32) * 0.2
c_host = np.ones(N_NEURONS, dtype=np.float32) * -65.0
d_host = np.ones(N_NEURONS, dtype=np.float32) * 8.0

# Assign Neuron Types (20% Inhibitory, others Excitatory)
types = np.random.randint(0, 100, N_NEURONS)
# Fast Spiking (Inhibitory)
fs_mask = (types < 20)
a_host[fs_mask] = 0.1; b_host[fs_mask] = 0.2; c_host[fs_mask] = -65.0; d_host[fs_mask] = 2.0
# Chattering
ch_mask = (types >= 20) & (types < 60)
a_host[ch_mask] = 0.02; b_host[ch_mask] = 0.2; c_host[ch_mask] = -50.0; d_host[ch_mask] = 2.0
# Intrinsically Bursting
ib_mask = (types >= 60) & (types < 80)
a_host[ib_mask] = 0.02; b_host[ib_mask] = 0.2; c_host[ib_mask] = -55.0; d_host[ib_mask] = 4.0

neuron_type_flags = np.zeros(N_NEURONS, dtype=np.int32)
neuron_type_flags[fs_mask] = 1 # 1: Inhibitory

# Connectivity (Random Sparse)
print("Generating connectivity map...")
post_indices_host = np.random.randint(0, N_NEURONS, (N_NEURONS, SYNAPSES_PER_NEURON)).astype(np.int32)
# Initial Weights (Magnitude)
weights_host = np.abs(np.random.randn(N_NEURONS, SYNAPSES_PER_NEURON).astype(np.float32) * 0.5)

# --- Transfer to GPU ---
print("Transferring to GPU...")
v_dev = cuda.to_device(v_host)
u_dev = cuda.to_device(u_host)
a_dev = cuda.to_device(a_host)
b_dev = cuda.to_device(b_host)
c_dev = cuda.to_device(c_host)
d_dev = cuda.to_device(d_host)
weights_dev = cuda.to_device(weights_host)
post_indices_dev = cuda.to_device(post_indices_host)
neuron_type_flags_dev = cuda.to_device(neuron_type_flags)

# GPU buffers
I_dev = cuda.device_array(N_NEURONS, dtype=np.float32)
fired_dev = cuda.device_array(N_NEURONS, dtype=np.int32)
trace_host = np.zeros((N_NEURONS, SYNAPSES_PER_NEURON), dtype=np.float32)
trace_dev = cuda.to_device(trace_host)

# RNG States
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
rng_states = create_xoroshiro128p_states(N_NEURONS, seed=42)

# --- CUDA Kernels ---

@cuda.jit
def update_neurons_kernel(v, u, I, fired, a, b, c, d, time_step, md_phase, ld_active, rng_states):
    idx = cuda.grid(1)
    if idx < v.shape[0]:
        dt = 0.5
        
        # Noise injection
        noise = cuda.random.xoroshiro128p_normal_float32(rng_states, idx) * 2.0
        
        # External Input Logic (Placeholder for task specific input)
        ext_I = 0.0
        if (idx % 10) >= 2 and (idx % 10) < 6: # Rhythm input
            theta = math.sin(time_step * 0.1 + md_phase) 
            if theta > 0.8: ext_I = 50.0 
        if (idx % 10) >= 6 and (idx % 10) < 8: # Sensory input
            if ld_active > 0.5:
                ext_I = 100.0
        
        current_I = I[idx] + noise + ext_I 
        
        # Izhikevich Equations
        curr_v = v[idx]
        curr_u = u[idx]
        pa = a[idx]; pb = b[idx]; pc = c[idx]; pd = d[idx]
        
        dv = 0.04 * curr_v**2 + 5.0 * curr_v + 140.0 - curr_u + current_I
        du = pa * (pb * curr_v - curr_u)
        
        curr_v += dv * dt
        curr_u += du * dt
        
        # Spike check
        is_fired = 0
        if curr_v >= 30.0:
            curr_v = pc
            curr_u += pd
            is_fired = 1
            
        v[idx] = curr_v
        u[idx] = curr_u
        fired[idx] = is_fired

@cuda.jit
def propagate_synapses_kernel(fired, weights, post_indices, I_out, trace, reward, neuron_types):
    idx = cuda.grid(1)
    if idx < fired.shape[0]:
        is_fired = fired[idx]
        is_inhibitory = neuron_types[idx]
        
        for i in range(SYNAPSES_PER_NEURON):
            # 1. Propagate Current
            if is_fired == 1:
                target_idx = post_indices[idx, i]
                w = weights[idx, i]
                # Apply sign based on neuron type (Excitatory vs Inhibitory)
                if is_inhibitory == 1:
                    cuda.atomic.add(I_out, target_idx, -w) 
                else:
                    cuda.atomic.add(I_out, target_idx, w)
            
            # 2. Update Trace (Eligibility Trace)
            tr = trace[idx, i]
            if is_fired == 1: tr = 1.0
            else: tr *= 0.95 # Decay
            
            # 3. Apply Plasticity (Reward-modulated)
            if reward != 0.0 and tr > 0.1:
                lr = 0.01 # Learning rate
                dw = reward * tr * lr
                
                new_w = weights[idx, i] + dw
                # Clip weights
                if new_w > 10.0: new_w = 10.0
                if new_w < 0.0: new_w = 0.0
                weights[idx, i] = new_w
            
            trace[idx, i] = tr

# --- Main Simulation Loop ---
threadsperblock = 256
blockspergrid = (N_NEURONS + (threadsperblock - 1)) // threadsperblock

history_reward = []
print("Starting simulation loop...")
start_total = time.time()

for epoch in range(EPOCHS):
    # Task: Sync (0) vs Async (1)
    target_timing = 0 if epoch % 2 == 0 else 1
    
    reward_sum = 0
    total_spikes = 0
    
    for t in range(STEPS_PER_EPOCH):
        # Reset input current buffer
        I_dev[:] = 0.0
        
        # Input Signal Logic
        theta = math.sin(t * 0.1)
        is_peak = theta > 0.5
        ld_active = 0.0
        
        if target_timing == 0: # Sync task
            if is_peak: ld_active = 1.0
        else: # Async task
            if not is_peak: ld_active = 1.0
            
        # Kernel 1: Update Neurons
        update_neurons_kernel[blockspergrid, threadsperblock](
            v_dev, u_dev, I_dev, fired_dev, a_dev, b_dev, c_dev, d_dev, float(t), 0.0, float(ld_active), rng_states
        )
        
        # Calculate Reward (Sparse check)
        current_reward = 0.0
        if t % 20 == 0:
            fired_host = fired_dev.copy_to_host()
            total_spikes += np.sum(fired_host)
            
            # Check output layer (last 1000 neurons)
            output_spikes = np.sum(fired_host[-1000:])
            
            # Reward Logic (Analog)
            if target_timing == 0: # Want high activity
                current_reward = min(1.0, output_spikes * 0.05)
                if output_spikes == 0: current_reward = -0.1
            else: # Want low activity
                current_reward = -min(1.0, output_spikes * 0.05)
                if output_spikes == 0: current_reward = 0.5
            
            reward_sum += current_reward
        
        # Kernel 2: Propagate & Learn
        propagate_synapses_kernel[blockspergrid, threadsperblock](
            fired_dev, weights_dev, post_indices_dev, I_dev, trace_dev, float(current_reward), neuron_type_flags_dev
        )

    avg_reward = reward_sum / (STEPS_PER_EPOCH / 20)
    history_reward.append(avg_reward)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Target={target_timing}, Reward={avg_reward:.3f}, Spikes={total_spikes}")

elapsed = time.time() - start_total
print(f"Simulation Complete. Total Time: {elapsed:.2f}s")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(history_reward, marker='o', markersize=2)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
print(f"Graph saved to {OUTPUT_DIR}/learning_curve.png")
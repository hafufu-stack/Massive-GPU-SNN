# Massive Spiking Neural Network on GPU

Simulating millions of neurons in real-time using Python, Numba, and CUDA.
Targeting high-end consumer GPUs like the **NVIDIA RTX 5080 Laptop GPU**.

## 🚀 Features
- **Scale:** Supports 5-6 million neurons and hundreds of millions of synapses.
- **Performance:** Runs extremely fast (approx. 0.5s - 1.0s per generation) on RTX 5080.
- **Model:** Izhikevich neuron model with 4 types (RS, FS, CH, IB).
- **Learning:** Reward-modulated STDP (Spike-Timing Dependent Plasticity).

## 🛠 Requirements
- Python 3.10+
- NVIDIA GPU (Compute Capability 8.0+)
- CUDA Toolkit
- `pip install numba numpy matplotlib scipy cupy`

## 📦 Usage
1. Clone this repository.
2. Run the script:
   ```bash
   python snn_gpu.py

## 📝 Note
This is an experimental project to test the limits of consumer hardware for neuromorphic computing.
Pull requests and ideas are welcome!

## 📜 License
MIT License

## 👤 Author
[note](https://note.com/cell_activation/m/m5bf070b82882)

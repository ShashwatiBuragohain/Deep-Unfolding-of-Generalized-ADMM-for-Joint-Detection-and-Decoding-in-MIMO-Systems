# Deep Unfolding of Generalized ADMM for Neural-Networks
Because classical algorithms are smart… but making them learn? That’s genius.”

Modern wireless systems (5G, Wi-Fi 7, Massive MIMO, etc.) transmit data using dozens of antennas simultaneously.
More antennas = faster speeds;but also extremely hard detection problem at the receiver.
The base station receives a messy mixture:
           y = Hx + n

where:
x = transmitted QAM symbols
H = channel matrix
y = noisy received signal

Our job is to recover x from y with near-ML accuracy but without ML-level complexity. That’s where this project comes in.

Some Classical detectors and why they fail:

**MMSE**
* Fast, but basically guesses.
* Fails badly in overloaded or high-interference MIMO.

**MAP / ML Detection**

Optimal but computationally difficult in large MIMO.

**ADMM**

A strong optimization-based detector. But it uses:
* fixed penalty parameters
* hard projections
* no learning
* slow or unstable convergence

So here comes Deep Unfolding in the picture

Instead of using a giant black-box neural network (which is overkill), we take a strong algorithm (ADMM/PS-ADMM) and unwrap its iterations into layers of a neural network.

Each ADMM iteration → 1 neural network layer

Each layer gets learnable parameters:
* penalty weights ρₗ
* projection smoothness
* PS-ADMM level weights αₗ

Everything remains structured. Everything becomes trainable. And it just works.

## PS-ADMM

We use a Bit-Plane Trick for QAM Optimization
QAM points like {±1, ±3} are discrete.

Discrete + optimization = pain.

So, we break each symbol into bit-plane components:

For 16-QAM:

       x = x0 + 2*x1
 
- `x0` → captures the LSB structure
- `x1` → captures the MSB structure  

Each component lives in the continuous interval: [-1, +1]

This makes the search space smoother and optimization far more stable.

### PS-ADMM Update Rule (Per Bit-Plane)

For each bit-plane `q`, PS-ADMM updates:
     x_q^{k+1} = ( 2^q / (4^q * ρ − α_q) ) * [ ρ( x0 − Σ_{p≠q} 2^p * x_p ) + y ]
Higher bit-planes (MSB) get bigger denominators → move more carefully.
Lower bit-planes (LSB) move faster.

This multi-resolution structure makes the updates extremely effective for discrete QAM optimization.
<img src="https://github.com/user-attachments/assets/2307a1c6-2802-40c3-8421-2028f2fde35b" width="400" />


## Unfolding PS-ADMM → A Smarter, Trainable Detector

We turn each iteration into a learnable layer:

Layer 1 → update (x0, xq, dual variables)
Layer 2 → update again but with different ρ2, α2
...
Layer L → final refined estimate


The algorithm remains interpretable, not a black box.

## System Diagram (Detector ↔ Decoder Loop)
![flowdiagram1 1_page-0001](https://github.com/user-attachments/assets/abc1ce26-7af0-4d28-b95c-974881f611e4)

The detector outputs soft LLRs, the LDPC decoder refines them and returns a priori info back to detector.

Yes; turbo style.

Detector → Soft Info → Decoder
Decoder → Extrinsic LLR → Detector
(repeat)


## BER vs SNR for all detectors

![Unfolded BER curves](https://github.com/user-attachments/assets/54edfa1c-326f-4af8-b8fc-c2bcc993309d)


We teach classical ADMM and PS-ADMM how to learn like a neural network while keeping their mathematical structure — resulting in a fast, stable, accurate MIMO detector that outperforms both optimization and deep learning alone.

So this project is interpretable, scalable, generalizes well, merges optimization + deep learning and beats MMSE & classical ADMM

It’s far simpler than black-box neural MIMO detectors


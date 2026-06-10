# **Geometric Neuron v8: The Continuous-Time Dynamic Architecture**

A fully self-organizing, continuous-time neural architecture based on delay-space geometry, topological state tracking, and Grassmannian gradient flows.  
Unlike standard feedforward or recurrent networks that rely on dense matrix multiplications and discrete time steps, v8 operates on a **sparse, derivative-driven delta-code** interacting with a **shared, norm-stabilized global field**.  
This repository represents the closure of the framework's core architectural problem: achieving native population diversity and signal coverage without relying on brittle heuristics or artificial penalties.

## **🧠 The Architecture: Closing the Read/Write Asymmetry**

The fundamental breakthrough of v8 is the realization that population orthogonalization and data coverage are solved purely by the geometry of the space the units inhabit. The system achieves a perfect symmetry between how it *writes* to the field and how it *reads* from it.

### **1\. The Write-Path: The Sphere-Tangent Projector**

* **The Mechanism:** Units inject their state into a shared global field $v$, which is continuously norm-stabilized onto a hypersphere: $s(t) \= v(t) / \\lVert v(t) \\rVert$.  
* **The Physics:** The Jacobian of this normalization acts as a tangent projector: $\\frac{1}{\\lVert v \\rVert}(I \- ss^{\\mathsf{T}})$.  
* **The Result:** If a direction in the field is already occupied by Unit A, the gradient for Unit B in that direction is annihilated. The population's write-vectors ($P\_k$) naturally divide labor and orthogonalize.

### **2\. The Read-Path: The Ky Fan Trace Objective (Stiefel Manifold)**

* **The Problem in prior versions:** Pushing read templates apart (via frame potentials) minimized mutual coherence but left the templates blind to the actual data manifold (coverage stagnated).  
* **The Mechanism:** v8 replaces artificial penalties with the **Ky Fan trace objective**. It maximizes $\\text{tr}(Q^{\\mathsf{H}} C Q)$, where $C$ is the event-sampled increment covariance of the field, and $Q$ is the orthonormalized frame of the read-templates.  
* **The Physics:** By constraining the templates to the Stiefel manifold (via differentiable QR retraction), they perform a continuous-time Oja's subspace flow.  
* **The Result:** The templates are pulled natively onto the principal transition modes of the field's dynamics. Coverage is maximized organically. Because it is a gradient flow on a compact manifold, LaSalle's invariance principle guarantees convergence without limit-cycle oscillations.

## **⚙️ Core Components**

* **The Delta-Code:** Units do not fire based on static position. They fire based on the *derivative* (the tangent increment) of the field matching their stored Koopman eigenfunction.  
* **Chiral Readouts (Optional but Native):** The system natively supports substituting the symmetric power covariance $C$ with the skew-symmetric lag covariance $H\_\\tau$. This naturally forces templates to cover the dominant **rotation planes** (traveling waves) of the sequence, perfectly aligning with the $L\_k \= \\text{Im}(z\_k(t) \\bar{z}\_k(t \- \\tau))$ angular momentum operator.

## **📊 Empirical Validation (The Ledger)**

The geometric\_neuron\_v8.py engine includes a head-to-head empirical validation of the read-path objective on a continuous-motion task.  
**The Decisive Metric: Captured-Energy Fraction**  
*How much of the data's dominant subspace do the templates actually span?*

| Arm | g-Coherence | Captured-Energy Fraction |
| :---- | :---- | :---- |
| **Baseline** | 0.193 | 0.322 |
| **Frame Potential (v7)** | 0.006 | 0.324 |
| **Ky Fan Coverage (v8)** | 0.148 | **0.505** |

* **Finding 1:** The frame potential is provably data-blind. It drops coherence to near-zero but fails to capture any additional energy from the signal.  
* **Finding 2:** The Ky Fan objective successfully pulls the templates onto the data's dominant subspace, raising captured energy by \+57% with no loss in tracking alignment.

## **🚀 Running the Engine**

Dependencies:

```Bash  
pip install torch numpy matplotlib
```

Run the v8 head-to-head test:

```Bash  
python geometric\_neuron\_v8.py
```

## **📜 The Ethos**

This architecture is built on the physical realities of dynamical systems, not hyperparameter hunting.  
*Do not hype. Do not lie. Just show.*

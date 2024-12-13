# Reproducing CP-PINNs: Data-Driven Changepoint Detection in PDEs

This repository implements the methodology described in the paper [**CP-PINNs: Data-Driven Changepoints Detection in PDEs Using Online Optimized Physics-Informed Neural Networks**](https://arxiv.org/abs/2208.08626). The goal is to detect changepoints in the parameters of partial differential equations (PDEs) using the architecture and algorithm of Physics-Informed Neural Networks (PINNs).

## Features

- Detects changepoints in PDE parameters effectively.
- Incorporates both data-driven and physics-based constraints for robust predictions.
- Uses state-of-the-art optimization techniques for improved training.

## Key Highlights
- **Advection-Diffusion Equation**: 
  $$\frac{\partial u(x,t)}{\partial t} + \frac{\partial u(x,t)}{\partial x} = \lambda(t) \frac{\partial^2 u(x,t)}{\partial x^2}$$
  where the parameter $\lambda(t)$ changes over time, introducing change points.

- **Goals**:
  - Estimate $\lambda(t)$ and identify the associated change points.
  - Use CP-PINNs to perform better than traditional PINNs in scenarios involving drastic parameter changes.

- **Loss Function**:
  - Residual of the PDE.
  - Training data and boundary conditions.
  - Regularization term for $\lambda(t)$ using total variation regularization.

- **Training Algorithms**:
- - **Neural Network Architecture**:  
  A Physics-Informed Neural Network (PINN) is used to approximate the solution of PDEs while incorporating prior knowledge from physics.

- **Weight Initialization**:  
  Xavier initialization is applied to initialize the neural network weights.

- **Optimization**:  
  We use two different optimizers sequentially to minimize the loss function:
  1. **Adam**: For faster convergence during the initial stages of training.
  2. **L-BFGS**: For fine-tuning and achieving a more precise solution.
     
  - Pre-training with ordinary PINNs for better initialization.
  - Then we use the following algorithms.
 
    ### Cost Function and Batch Learning Description

In this implementation, we optimize a total cost function for a neural network model during batch learning. The total cost function is defined as:
$$\[
L(\mathbf{w};\boldsymbol{\Theta},\lambda(t)) = w_1 L^{NN} + w_2 L^{Training} + w_3 V^{\mathfrak{\lambda}}
\]$$

where:
- $\( w_1, w_2, w_3 \)$ are weights that control the relative importance of the terms in the cost function.
- $\( L^{NN} \)$: Neural network loss.
- $\( L^{Training} \)$: Loss from the training dataset.
- $\( V^{\mathfrak{\lambda}} \)$: A regularization or constraint term for \( \lambda(t) \).

The constraint $\( w_1 + w_2 + w_3 = 1 \)$ ensures proper normalization of the weights.

#### Optimization Workflow
1. **Minimizing the Cost Function:**
   - For the $\((k-1)^{th}\)$ batch, the cost function $\( L \)$ is minimized with respect to the neural network weights $\( \boldsymbol{\Theta} \)$.

2. **Updating Weights $\( \mathbf{w} \)$:**
   - After optimizing $\( \boldsymbol{\Theta} \)$, the weights $\( \mathbf{w} \)$ are updated for the $\(k^{th}\)$ batch using the formula:

     $\[\begin{bmatrix}
     w_1^{(k)} \\
     w_2^{(k)} \\
     w_3^{(k)}\end{bmatrix} = \begin{bmatrix}
     \exp\left[-\eta  L^{NN}_{(k-1)} - \left( 1-\eta\gamma\right)\right] \\
     \exp\left[-\eta L^{Training}_{(k-1)} - \left( 1-\eta\gamma\right)\right] \\
     \exp\left[-\eta V^{\mathfrak{\lambda}}_{(k-1)} - \left( 1-\eta\gamma\right)\right]
     \end{bmatrix}\]$

   where:
   - $\( \eta \)$: Learning rate.
   - $\( \gamma \)$: Regularization parameter.
   - $\( L^{NN}_{(k-1)} \), \( L^{Training}_{(k-1)} \), and \( V^{\mathfrak{\lambda}}_{(k-1)} \)$: Respective loss values from the $\((k-1)^{th}\)$ batch.

### Implementation Notes
- The weights $\( \mathbf{w} \)$ are re-normalized after each update to ensure their sum remains 1.
- This dynamic weighting mechanism adapts the importance of each term based on the losses from the previous batch, improving the model's robustness and flexibility during training.


## Results
- **Change Points**: $t_1 = 0.33 \pm 0.01$, $t_2 = 0.66 \pm 0.01$.
- **Estimated Parameters**:
  - $\lambda_1 = 0.5 \pm 0.0017$.
  - $\lambda_2 = 0.05 \pm 0.0011$.
  - $\lambda_3 = 1.0 \pm 0.0007$.

## Applications
- Detecting change points in physical systems governed by PDEs.
- Quantitative finance: Identifying points of high volatility.
- Other domains requiring parameter estimation in time-dependent PDEs.

## Files
- **`final.ipynb`**: Main jupyter notebook source code.
- **Figures**:
  - `Numerical_Solution.png`: Numerical solution of the advection-diffusion equation.
  - `Training Data.png`: Training data visualization.
  - `Lambda_plot.png`: Estimated $\lambda(t)$ after CP-PINN training.
  - `Absolute Difference.png`: Absolute error between training data and the neural network solution.

- CP-PINNs excel in solving PDEs with time-dependent parameters and detecting change points.
- Applications include quantitative finance, e.g., estimating high-volatility points.

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
  - Pre-training with ordinary PINNs for better initialization.
  - Gradient-based optimization with dynamically updated loss weights.
  - 
Our implementation follows the core components of the CP-PINNs framework:

- **Neural Network Architecture**:  
  A Physics-Informed Neural Network (PINN) is used to approximate the solution of PDEs while incorporating prior knowledge from physics.

- **Weight Initialization**:  
  Xavier initialization is applied to initialize the neural network weights.

- **Optimization**:  
  We use two different optimizers sequentially to minimize the loss function:
  1. **Adam**: For faster convergence during the initial stages of training.
  2. **L-BFGS**: For fine-tuning and achieving a more precise solution.

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

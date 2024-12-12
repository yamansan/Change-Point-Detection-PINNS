# Reproducing CP-PINNs: Data-Driven Changepoint Detection in PDEs

This repository implements the methodology described in the paper [**CP-PINNs: Data-Driven Changepoints Detection in PDEs Using Online Optimized Physics-Informed Neural Networks**](https://arxiv.org/abs/2208.08626). The goal is to detect changepoints in the parameters of partial differential equations (PDEs) using the architecture and algorithm of Physics-Informed Neural Networks (PINNs).

## Overview

Our implementation follows the core components of the CP-PINNs framework:

- **Neural Network Architecture**:  
  A Physics-Informed Neural Network (PINN) is used to approximate the solution of PDEs while incorporating prior knowledge from physics.

- **Weight Initialization**:  
  Xavier initialization is applied to initialize the neural network weights.

- **Optimization**:  
  We use two different optimizers sequentially to minimize the loss function:
  1. **Adam**: For faster convergence during the initial stages of training.
  2. **L-BFGS**: For fine-tuning and achieving a more precise solution.

- **Loss Function**:  
  The loss function comprises three terms:
  1. Mean Squared Error (MSE) from the training data.
  2. The squared residual of the governing PDE.
  3. A regularization term for the model parameters.

## Features

- Detects changepoints in PDE parameters effectively.
- Incorporates both data-driven and physics-based constraints for robust predictions.
- Uses state-of-the-art optimization techniques for improved training.


# Change Point - Physics Informed Neural Networks (CP-PINNs)

This repository contains a LaTeX-based presentation titled **Change Point - Physics Informed Neural Networks (CP-PINNs)**. The slides cover the use of CP-PINNs to estimate change points and time-dependent parameters in the advection-diffusion equation.

## Contents

1. [Title and Overview](#title-and-overview)
2. [Recap](#recap)
3. [Goal](#goal)
4. [Training Data](#training-data)
5. [Loss Function for CP-PINN](#loss-function-for-cp-pinn)
6. [Training Algorithm for CP-PINNs](#training-algorithm-for-cp-pinns)
7. [Using Ordinary PINNs for Pre-Training](#using-ordinary-pinns-for-pre-training)
8. [Training the Model with Pre-Trained Parameters](#training-the-model-with-pre-trained-parameters)
9. [Final Results](#final-results)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Title and Overview

### Slide: Title Page

The presentation introduces CP-PINNs, a neural network-based framework for solving PDEs with time-dependent parameters and detecting change points.

---

## Recap

### Slide: Recap

- Neural networks can approximate almost all practically useful functions (Universal Approximation Theorem).
- They can also approximate solutions to partial differential equations (PDEs).

---

## Goal

### Slide: Goal

- Solve the advection-diffusion equation:

  \[\frac{\partial u(x,t)}{\partial t} + \frac{\partial u(x,t)}{\partial x} = \lambda(t) \frac{\partial^2 u(x,t)}{\partial x^2}\]

  where:

  \[\lambda(t) = \begin{cases} 
  0.5 & \text{for } t \in [0, \frac{1}{3}), \\
  0.05 & \text{for } t \in [\frac{1}{3}, \frac{2}{3}), \\
  1.0 & \text{for } t \in [\frac{2}{3}, 1).
  \end{cases}\]

- Estimate \( \lambda(t) \) and change points \( t_1, t_2 \).

---

## Training Data

### Slide: Numerical Solution

Numerical solution to the advection-diffusion equation:

![Numerical Solution](images/Numerical_Solution.png)

### Slide: Training Data

Data points chosen from the numerical solution:

![Training Data](images/Training_Data.png)

---

## Loss Function for CP-PINN

### Slide: Loss Function

The total loss function includes three terms:

1. Residual of the PDE:
   \[L^{NN} = \sum_{i,j} \left( \frac{\partial u_{NN}}{\partial t} + \frac{\partial u_{NN}}{\partial x} - \lambda_{NN} \frac{\partial^2 u_{NN}}{\partial x^2} \right)^2 \]

2. Training data and boundary conditions:
   \[L^{Training} = \sum_{i,j} \left( u_{NN} - u_{train} \right)^2 \]

3. Regularization term for \( \lambda(t) \):
   \[V^{\mathfrak{\lambda}} = \sum_{i=1}^{T-1} \delta(t^i)\left|\Delta{\lambda}(t^i)\right|\]

---

## Training Algorithm for CP-PINNs

### Slide: Training Algorithm

- The total cost function:

  \[L(\mathbf{w};\boldsymbol{\Theta},\lambda(t)) = w_1 L^{NN}+ w_2 L^{Training}+ w_3 V^{\mathfrak{\lambda}}\]

- Update weights \( \mathbf{w} \) for each batch:

  \[\left[ 
  \begin{array}{c} 
  w_1^{(k)} \\
  w_2^{(k)} \\
  w_3^{(k)}
  \end{array} 
  \right] = \left[ 
  \begin{array}{c} 
  \exp\left[-\eta  L^{NN}_{(k-1)} - \left( 1-\eta\gamma\right)\right] \\
  \exp\left[-\eta L^{Training}_{(k-1)} - \left( 1-\eta\gamma\right)\right]\\
  \exp\left[-\eta V^{\mathfrak{{\hat{\lambda}}}}_{(k-1)} - \left( 1-\eta\gamma\right)\right]
  \end{array} 
  \right]\]

---

## Using Ordinary PINNs for Pre-Training

### Slide: Linear Regression Estimate

Estimate \( \lambda(t) \) and change points using linear regression:

![Lambda Plot Linear Regression](images/Lambda_plot_linear_regression.png)

---

## Training the Model with Pre-Trained Parameters

### Slide: CP-PINN Training

Using CP-PINN training:

![Lambda Plot](images/Lambda_plot.png)

### Slide: Absolute Difference

Absolute error between training data and neural network solution:

![Absolute Difference](images/Absolute_Difference.png)

---

## Final Results

### Slide: Results

- Change points: \( t_1 = 0.33 \pm 0.01 \), \( t_2 = 0.66 \pm 0.01 \).
- Values of \( \lambda \):
  - \( \lambda_1 = 0.5 \pm 0.0017 \)
  - \( \lambda_2 = 0.05 \pm 0.0011 \)
  - \( \lambda_3 = 1.0 \pm 0.0007 \)

---

## Summary and Next Steps

### Slide: Summary

- CP-PINNs excel in solving PDEs with time-dependent parameters and detecting change points.
- Applications include quantitative finance, e.g., estimating high-volatility points.

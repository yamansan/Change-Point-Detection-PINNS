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


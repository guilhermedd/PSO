# Particle Swarm Optimization (PSO) — Benchmark Functions

## Overview

**Particle Swarm Optimization (PSO)** is a population-based optimization algorithm inspired by the social behavior of birds and fish.  
Each **particle** represents a candidate solution and moves through the search space, guided by:
- Its own best-known position (**personal best**, or `pbest`),
- The best-known position found by the entire swarm (**global best**, or `gbest`).

The objective of PSO is to **minimize** a given function — in this project, the benchmark functions **Griewank** and **Ackley**.

---

## Benchmark Functions

### 1. Griewank Function

- **Equation**:
  \[
  f(x) = 1 + \frac{1}{4000} \sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right)
  \]
- **Properties**:
  - Global minimum at \( x_i = 0 \) for all \( i \).
  - \( f(x) = 0 \) at the global minimum.
- **Search Space**:
  - Each variable \( x_i \in [-600, 600] \).

### 2. Ackley Function

- **Equation**:
  \[
  f(x) = -20 \exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e
  \]
- **Properties**:
  - Global minimum at \( x_i = 0 \) for all \( i \).
  - \( f(x) = 0 \) at the global minimum.
- **Search Space**:
  - Each variable \( x_i \in [-32, 32] \).

---

## How PSO Works

1. **Initialization**:
   - Randomly generate positions and velocities for each particle within the specified domain.
   
2. **Evaluation**:
   - Compute the fitness of each particle using the selected benchmark function.
   
3. **Update**:
   - For each particle, update its velocity and position according to:
     ```text
     velocity = w * velocity + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)
     position = position + velocity
     ```
   - Where:
     - `w` is the inertia weight,
     - `c1` and `c2` are acceleration coefficients,
     - `r1` and `r2` are random values in [0, 1].

4. **Loop**:
   - Repeat the evaluation and update steps until a stopping condition is met (e.g., maximum number of iterations or satisfactory error threshold).
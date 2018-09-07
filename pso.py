#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def schwefel(X):
    """
    Shwefel function
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    
    Returns:
    Y -- The output of the function for input X
    
    """
    
    M, N = X.shape
    Y = 418.982887*N - np.sum(X*np.sin(np.sqrt(np.absolute(X))), axis=1, keepdims=True)
    assert(Y.shape == (M,1))
    return Y

def fitness(X):
    """
    Fitness function to find the minima of the Schwefel function
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    
    Returns:
    fitness -- Fitness scores with shape (M,1)
    
    """
    
    epsilon = 1e-100
    fitness = 1/(schwefel(X)+epsilon)
    return fitness

def absorbing_walls(X, boundaries):
    """
    Implementation of the absorbing walls boundary checking
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    boundaries -- boundary conditions, shape (N,2)
    
    Returns:
    X -- a batch of M particles, shape (M,N)
    
    """
    
    for j in range(X.shape[1]):
        columns_crossed_dim_j_lower = np.nonzero(X[:,j] < boundaries[j,0])[0]
        X[columns_crossed_dim_j_lower, j] = boundaries[j,0]*np.ones(columns_crossed_dim_j_lower.shape[0])
        columns_crossed_dim_j_upper = np.nonzero(X[:,j] > boundaries[j,1])[0]
        X[columns_crossed_dim_j_upper, j] = boundaries[j,1]*np.ones(columns_crossed_dim_j_upper.shape[0])
    return X

def reflecting_walls(X, boundaries):
    """
    Implementation of the reflecting walls boundary checking
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    boundaries -- boundary conditions, shape (N,2)
    
    Returns:
    X -- a batch of M particles, shape (M,N)
    
    """
    
    for j in range(X.shape[1]):
        # Check the particles (rows) that crossed the lower boundary in dimension j
        particles_crossed_dim_j_lower = np.nonzero(X[:,j] < boundaries[j,0])[0]
        while (particles_crossed_dim_j_lower.shape[0]!=0):
            X[particles_crossed_dim_j_lower, j] = 2*boundaries[j,0] - X[particles_crossed_dim_j_lower, j]
            particles_crossed_dim_j_lower = np.nonzero(X[:,j] < boundaries[j,0])[0]
        # Check the particles (rows) that crossed the upper boundary in dimension j
        particles_crossed_dim_j_upper = np.nonzero(X[:,j] > boundaries[j,1])[0]
        while (particles_crossed_dim_j_upper.shape[0]!=0):
            X[particles_crossed_dim_j_upper, j] = 2*boundaries[j,1] - X[particles_crossed_dim_j_upper, j]
            particles_crossed_dim_j_upper = np.nonzero(X[:,j] > boundaries[j,1])[0]
    return X

def invisible_walls(X, boundaries):
    """
    Implementation of the invisible walls boundary checking
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    boundaries -- boundary conditions, shape (N,2)
    
    Returns:
    X -- a batch of M particles, shape (M,N)
    
    """
    
    for j in range(X.shape[1]):
        particles_crossed = np.nonzero(np.logical_or(X < boundaries[j,0], X > boundaries[j,1]))[0]
        X[particles_crossed] = np.random.rand(particles_crossed.shape[0],X.shape[1])*(boundaries[j,1]-boundaries[j,0])+boundaries[j,0]
    return X

def initializePSO(M, N):
    """
    Initializes the parameters
    
    Arguments:
    M -- number of particles
    N -- dimensionality of the particle
    
    Returns:
    X -- a batch of M particles, shape (M,N)
    pbest_particles -- initial personal bests, shape (M,N)
    pbest_fitness -- initial personal best scores, shape (M,1)
    gbest_particle -- initial global best, shape (1,N)
    gbest_fitness -- initial global best score, integer
    boundaries -- boundary conditions, shape (N,2)
    V -- initial velocities, shape (M,N)
    
    """
    
    boundaries = 512*np.ones((N,2)) # for each of the N dimensions, a lower and an upper boundary
    boundaries[:,0] = -1*boundaries[:,0]
    X = np.random.rand(M,N)*(boundaries[0,1]-boundaries[0,0])+boundaries[0,0]
    pbest_particles = X
    pbest_fitness = fitness(X)
    gbest_fitness = pbest_fitness[np.argmax(pbest_fitness, axis=0)]
    gbest_particle = np.zeros((1,N))
    V = np.zeros((M,N))
    
    return X, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V

def runPSO(X, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V, I, M, N, w, c1, c2, dt, boundary_condition):
    """
    Runs PSO for a number of iterations
    
    Arguments:
    X -- a batch of M particles, shape (M,N)
    pbest_particles -- initial personal bests, shape (M,N)
    pbest_fitness -- initial personal best scores, shape (M,1)
    gbest_particle -- initial global best, shape (1,N)
    gbest_fitness -- initial global best score, integer
    boundaries -- boundary conditions, shape (N,2)
    V -- initial velocities, shape (M,N)
    I -- number of iterations -- termination criteria
    M -- number of particles
    N -- dimensionality of the particle
    w -- intertial weight
    c1 -- memory influence scaling coefficient
    c2 -- social influence scaling coefficient
    dt -- velocity time step
    boundary_condition -- upper and lower bounds
    
    Returns:
    gbest_particle -- global best particle
    gbest_fitness -- fitness score achieved by the global best particle
    
    """
    
    
    for i in range(I):
        # Evaluate current particles' fitness
        current_fitness = fitness(X)
        
        # Find the new personal best fitness and deduce the new global best fitness
        pbest_fitness_prev = pbest_fitness
        pbest_fitness = np.maximum(current_fitness, pbest_fitness_prev)
        gbest_fitness = pbest_fitness[np.argmax(pbest_fitness, axis=0)]
        
        # Update personal best particles and deduce the new global best particle
        particles_left = np.in1d(pbest_fitness, pbest_fitness_prev) # get the indices of the particles for which the pbest_fitness was unchanged (i.e. no better solution found)
        pbest_particles[particles_left, :] = X[particles_left, :] # update the pbest particles with the ones that outperformed their previous pbest
        gbest_particle = pbest_particles[np.argmax(pbest_fitness, axis=0),:]
        
        # Update the velocities of the particles
        V = w*V + c1*np.random.rand(M,N)*(pbest_particles - X) + c2*np.random.rand(M,N)*(gbest_particle - X)
        #V = np.random.rand(M,N) # Random particle movement, not recommended
        
        # Move the particles
        X = X + dt*V
        
        # Check boundary conditions
        if (boundary_condition == "absorbing_walls"):
            X = absorbing_walls(X, boundaries)
        elif (boundary_condition == "reflecting_walls"):
            X = reflecting_walls(X, boundaries)
        elif (boundary_condition == "invisible_walls"):
            X = invisible_walls(X, boundaries)
    
    return gbest_particle, gbest_fitness


if __name__ == "__main__":
    # Initialization
    X, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V = initializePSO(M=1000, N=2)
    
    # Run iterations
    gbest_particle, gbest_fitness = runPSO(X, pbest_particles, pbest_fitness, gbest_particle, gbest_fitness, boundaries, V, I=1000, M=1000, N=2, w=0.1, c1=0.1, c2=0.1, dt=0.1, boundary_condition="absorbing_walls")
    print('Global best found at: ', gbest_particle)
    print('The function value at global optima is: ', 1/gbest_fitness)
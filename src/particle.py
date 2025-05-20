import numpy as np

class Particle:
    def __init__(self, dimensions, bounds, c1, c2, vmax_ratio):
        self.dimensions = dimensions
        self.bounds = bounds
        self.c1 = c1
        self.c2 = c2
        self.vmax = (bounds[1] - bounds[0]) * vmax_ratio

        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-self.vmax, self.vmax, dimensions)

        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')

    def evaluate(self, objective_function):
        fitness = objective_function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = np.copy(self.position)
        return fitness, self.position

    def update_velocity(self, best_position, w=1.0):
        # best_position pode ser o global_best (PSO/PSOw) ou o melhor vizinho (PSOk)
        r1 = np.random.rand(self.dimensions)
        r2 = np.random.rand(self.dimensions)

        cognitive = self.c1 * r1 * (self.best_position - self.position)
        social = self.c2 * r2 * (best_position - self.position)
        inertia = w * self.velocity

        self.velocity = inertia + cognitive + social
        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

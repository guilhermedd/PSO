from typing import Callable, Tuple
import numpy as np
import random

class Particle:
    def __init__(
        self,
        dimension: int,
        bounds: Tuple[float, float],
        c1: float = 2.05,
        c2: float = 2.05,
        vmax_ratio: float = 0.2
    ):
        """
        Inicializa uma partícula para o algoritmo PSO.
        
        Args:
            dimension: Dimensão do espaço de busca
            bounds: Tupla com os limites inferior e superior do espaço de busca
            c1: Coeficiente cognitivo
            c2: Coeficiente social
            vmax_ratio: Razão do intervalo de busca para definir velocidade máxima
        """
        self.dimension = dimension
        self.bounds = bounds
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax_ratio * (bounds[1] - bounds[0])
        
        # Constante de constrição
        theta = c1 + c2
        self.constriction = 2 / (2 - theta - np.sqrt(theta**2 - 4 * theta)) if theta > 4 else 1
        
        # Inicializa posição e velocidade aleatoriamente dentro dos limites
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(dimension)])
        self.velocity = np.array([random.uniform(-self.vmax, self.vmax) for _ in range(dimension)])
        
        # Melhores valores locais
        self.best_fitness = float('inf')
        self.best_position = np.copy(self.position)
        self.current_fitness = float('inf')

    def update_velocity(self, global_best_position: np.ndarray) -> None:
        """Atualiza a velocidade da partícula usando o melhor global."""
        r1 = np.random.random(self.dimension)
        r2 = np.random.random(self.dimension)
        
        cognitive = self.c1 * r1 * (self.best_position - self.position)
        social = self.c2 * r2 * (global_best_position - self.position)
        
        new_velocity = self.constriction * (self.velocity + cognitive + social)
        
        # Limita a velocidade
        self.velocity = np.clip(new_velocity, -self.vmax, self.vmax)

    def update_position(self) -> None:
        """Atualiza a posição da partícula e aplica os limites de busca."""
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def evaluate(self, objective_function: Callable[[np.ndarray], float]) -> Tuple[float, np.ndarray]:
        """Avalia a partícula e atualiza seu melhor local."""
        self.current_fitness = objective_function(self.position)
        
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_position = np.copy(self.position)
            
        return self.current_fitness, self.position
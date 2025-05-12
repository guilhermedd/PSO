from typing import Callable, Tuple, List
from src.particle import Particle
import numpy as np
from dataclasses import dataclass

@dataclass
class PSOResult:
    best_fitness: float
    best_position: np.ndarray
    iterations: int
    fitness_history: List[float]

class PSOEnvironment:
    def __init__(
        self,
        dimensions: int,
        bounds: Tuple[float, float],
        objective_function: Callable[[np.ndarray], float],
        num_particles: int = 100,   
        c1: float = 1.5,            
        c2: float = 1.5,            
        vmax_ratio: float = 0.1     
    ):
        """
        Ambiente para execução do algoritmo PSO.
        
        Args:
            dimensions: Dimensão do problema
            bounds: Limites inferior e superior do espaço de busca
            objective_function: Função objetivo a ser minimizada
            num_particles: Número de partículas no enxame
            c1: Coeficiente cognitivo
            c2: Coeficiente social
            vmax_ratio: Razão do intervalo de busca para definir velocidade máxima
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.vmax_ratio = vmax_ratio
        
        self.global_best_position = np.zeros(dimensions)
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        
        # Inicializa o enxame de partículas
        self.particles = [
            Particle(dimensions, bounds, c1, c2, vmax_ratio)
            for _ in range(num_particles)
        ]

    def optimize(
        self,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> PSOResult:
        """
        Executa a otimização por PSO.
        
        Args:
            max_iterations: Número máximo de iterações
            tolerance: Valor de tolerância para convergência
            stagnation_limit: Número de iterações sem melhoria para parada antecipada
            verbose: Se True, imprime progresso durante a execução
            
        Returns:
            PSOResult: Objeto com os resultados da otimização
        """
        iterations = 0
        
        while (self.global_best_fitness > tolerance):
            
            iteration_fitness = []
            
            # Avalia todas as partículas
            for particle in self.particles:
                fitness, position = particle.evaluate(self.objective_function)
                iteration_fitness.append(fitness)
                
                # Atualiza melhor global se necessário
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(position)
                    best_iteration = iterations
            
            # Atualiza velocidades e posições
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
            
            # Registra histórico e verifica estagnação
            self.fitness_history.append(self.global_best_fitness)
            
            iterations += 1
            
            # Log de progresso
            if verbose and (iterations % 100 == 0 or iterations == 1):
                print(f"Iteration {iterations:5d} - Best Fitness: {self.global_best_fitness:.6f}")
        
        return PSOResult(
            best_fitness=self.global_best_fitness,
            best_position=self.global_best_position,
            iterations=iterations,
            fitness_history=self.fitness_history
        )
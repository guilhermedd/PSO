from src.environment import PSOEnvironment
import numpy as np

def griewank(solution: np.ndarray) -> float:
    """Função Griewank - intervalo recomendado: [-600, 600]"""
    dim = len(solution)
    sum_part = np.sum(solution**2) / 4000
    product_part = np.prod(np.cos(solution / np.sqrt(np.arange(1, dim + 1))))
    return sum_part - product_part + 1

def ackley(solution: np.ndarray) -> float:
    """Função Ackley - intervalo recomendado: [-32.768, 32.768]"""
    dim = len(solution)
    sum_sq = np.sum(solution**2)
    sum_cos = np.sum(np.cos(2 * np.pi * solution))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    
    return term1 + term2 + 20 + np.e

if __name__ == '__main__':
    # Configurações otimizadas para Ackley
    TEST_FUNCTION = "griewank"
    DIMENSIONS = 10
    NUM_PARTICLES = 500  # Enxame maior
    
    if TEST_FUNCTION == "ackley":
        BOUNDS = (-32.768, 32.768)
        OBJECTIVE_FUNCTION = ackley
        C1 = 1.5
        C2 = 1.5
        VMAX_RATIO = 0.1  
    else:
        BOUNDS = (-600, 600)
        OBJECTIVE_FUNCTION = griewank
        C1 = 1.5
        C2 = 1.5
        VMAX_RATIO = 0.1  
    
    pso = PSOEnvironment(
        dimensions=DIMENSIONS,
        bounds=BOUNDS,
        objective_function=OBJECTIVE_FUNCTION,
        num_particles=NUM_PARTICLES,
        c1=C1,
        c2=C2,
        vmax_ratio=VMAX_RATIO
    )
    
    result = pso.optimize(
        tolerance=1e-10,  # Tolerância mais rigorosa
        verbose=True
    )
    
    # Exibe resultados
    print("\nOptimization Results:")
    print(f"Best fitness: {result.best_fitness}")
    print(f"Best position: {result.best_position}")
    print(f"Iterations: {result.iterations}")
    print(f"Final fitness: {result.fitness_history[-1]}")
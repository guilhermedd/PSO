from src.environment import PSOEnvironment
import numpy as np
import matplotlib.pyplot as plt

def griewank(solution: np.ndarray) -> float:
    dim = len(solution)
    sum_part = np.sum(solution**2) / 4000
    product_part = np.prod(np.cos(solution / np.sqrt(np.arange(1, dim + 1))))
    return sum_part - product_part + 1

def ackley(solution: np.ndarray) -> float:
    dim = len(solution)
    sum_sq = np.sum(solution**2)
    sum_cos = np.sum(np.cos(2 * np.pi * solution))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    return term1 + term2 + 20 + np.e

if __name__ == '__main__':
    TEST_FUNCTION = "griewank"
    NUM_PARTICLES = 30
    DIMENSIONS = [5, 10]

    for dim in DIMENSIONS:
        fits = []
        max_len = 0

        for i in range(10):
            if TEST_FUNCTION == "ackley":
                BOUNDS = (-32.768, 32.768)
                OBJECTIVE_FUNCTION = ackley
            else:
                BOUNDS = (-600, 600)
                OBJECTIVE_FUNCTION = griewank

            pso = PSOEnvironment(
                dimensions=dim,
                bounds=BOUNDS,
                objective_function=OBJECTIVE_FUNCTION,
                num_particles=NUM_PARTICLES,
                c1=2.05,
                c2=2.05,
                vmax_ratio=0.1
            )

            assert pso.c1 + pso.c2 > 4, "c1 + c2 deve ser maior que 4"

            result = pso.optimize(tolerance=1e-10, verbose=True)
            fits.append(result.fitness_history)
            max_len = max(max_len, len(result.fitness_history))

        # Preenchimento com NaN para alinhamento
        fits_padded = [np.pad(f, (0, max_len - len(f)), constant_values=np.nan) for f in fits]
        mean_fit = np.nanmean(fits_padded, axis=0)

        plt.plot(mean_fit)
        plt.title(f"Convergência do PSO - {TEST_FUNCTION} ({dim}D, {NUM_PARTICLES} partículas)")
        plt.xlabel("Iterações")
        plt.ylabel("Fitness")
        plt.yscale("log")
        plt.grid()
        plt.savefig(f"graps/convergencia_{TEST_FUNCTION}_{dim}D.png")
        plt.show()

        print("\nOptimization Results:")
        print(f"Best fitness: {result.best_fitness}")
        print(f"Best position: {result.best_position}")
        print(f"Iterations: {result.iterations}")
        print(f"Final fitness: {result.fitness_history[-1]}")
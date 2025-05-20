from src.environment import PSOEnvironment
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def griewank(solution: np.ndarray) -> float:
    dim = len(solution)
    sum_part = np.sum(solution**2) / 4000
    angles = (solution / np.sqrt(np.arange(1, dim + 1))) * np.pi / 180
    product_part = np.prod(np.cos(angles))
    return sum_part - product_part + 1


def ackley(solution: np.ndarray) -> float:
    dim = len(solution)
    sum_sq = np.sum(solution**2)
    sum_cos = np.sum(np.cos(2 * np.pi * solution))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    return term1 + term2 + 20 + np.e


# Função para ser executada em paralelo
def run_pso(dim, bounds, obj_fn, num_particles, variant, w=0.729, k=3):
    pso = PSOEnvironment(
        dimensions=dim,
        bounds=bounds,
        objective_function=obj_fn,
        num_particles=num_particles,
        c1=2.05,
        c2=2.05,
        vmax_ratio=0.1,
        variant=variant,
        w=w,
        k=k
    )
    result = pso.optimize()
    return result.fitness_history, result.best_fitness, result.best_position, result.iterations


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")  # mais seguro, especialmente no Windows/macOS

    TEST_FUNCTIONS = ["ackley", "griewank"]
    NUM_PARTICLES = 30
    DIMENSIONS = [5, 10]
    VARIANTS = [
        ("standard", {"w": 0.729, "k": 3}),
        ("w", {"w": 0.729, "k": 3}),
        ("k", {"w": 1.0, "k": 3}),
    ]

    for test_function in TEST_FUNCTIONS:
        if test_function == "ackley":
            BOUNDS = (-32.768, 32.768)
            OBJ_FN = ackley
        else:
            BOUNDS = (-600, 600)
            OBJ_FN = griewank

        for dim in DIMENSIONS:
            for variant, params in VARIANTS:
                print(f"Executando {test_function} com {dim} dimensões, variante {variant}...")

                with multiprocessing.Pool(processes=10) as pool:
                    results = pool.starmap(
                        run_pso,
                        [(dim, BOUNDS, OBJ_FN, NUM_PARTICLES, variant, params["w"], params["k"]) for _ in range(10)]
                    )

                fits = [r[0] for r in results]
                best_fitnesses = [r[1] for r in results]
                best_positions = [r[2] for r in results]
                iterations_list = [r[3] for r in results]

                max_len = max(len(fit) for fit in fits)
                fits_padded = [np.pad(f, (0, max_len - len(f)), constant_values=np.nan) for f in fits]
                mean_fit = np.nanmean(fits_padded, axis=0)

                # Resultados
                with open(f"results/{test_function}_{dim}D_{variant}.txt", "w") as f:
                    f.write(f"Desvio padrão: {np.std(mean_fit)}\n")
                    f.write(f"Fit médio: {np.mean(mean_fit)}\n")

                # Gráfico de convergência
                plt.plot(mean_fit)
                plt.title(f"Convergência do PSO-{variant} - {test_function} ({dim}D, {NUM_PARTICLES} partículas)")
                plt.xlabel("Iterações")
                plt.ylabel("Fitness")
                plt.yscale("log")
                plt.grid()
                plt.savefig(f"graphs/convergencia_{test_function}_{dim}D_{variant}.png")
                plt.close()

                # Boxplot
                final_fitness = [f[-1] for f in fits]
                plt.boxplot(final_fitness)
                plt.title(f"Boxplot de Fitness Final - {test_function} ({dim}D, {NUM_PARTICLES} partículas, {variant})")
                plt.ylabel("Fitness final")
                plt.grid()
                plt.yscale("log")
                plt.savefig(f"graphs/boxplot_{test_function}_{dim}D_{variant}.png")
                plt.close()

                print("\nOptimization Results:")
                print(f"Best fitness: {min(best_fitnesses)}")
                print(f"Best position: {best_positions[np.argmin(best_fitnesses)]}")
                print(f"Iterations: {iterations_list[np.argmin(best_fitnesses)]}")
                print(f"Final fitness: {final_fitness[np.argmin(best_fitnesses)]}")

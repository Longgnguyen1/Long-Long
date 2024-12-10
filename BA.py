import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def objective_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def bat_algorithm_3d_multiple_solutions(obj_func, num_bats, max_iter, dim, x_bounds, f_min, f_max, A, r_0, alpha=0.9, gamma=0.9):
    bats_position = np.random.rand(num_bats, dim) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    bats_velocity = np.random.rand(num_bats, dim) * 0.01
    bats_frequency = f_min + (f_max - f_min) * np.random.rand(num_bats)
    bats_loudness = np.ones(num_bats) * r_0
    bats_pulse_rate = np.random.rand(num_bats)


    fitness = np.apply_along_axis(obj_func, 1, bats_position)
    best_solutions = [bats_position[np.argmin(fitness)]]
    best_fitness = [np.min(fitness)]

    def is_new_solution(candidate, solutions, threshold=0.5):
        for solution in solutions:
            if np.linalg.norm(candidate - solution) < threshold:
                return False
        return True

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for iter in range(max_iter):
        for i in range(num_bats):
    
            bats_frequency[i] = f_min + (f_max - f_min) * np.random.rand()
            bats_velocity[i] += (bats_position[i] - best_solutions[0]) * bats_frequency[i]
            bats_position[i] += bats_velocity[i]
            bats_position[i] = np.clip(bats_position[i], x_bounds[0], x_bounds[1])
            r = np.random.rand()
            if r < bats_pulse_rate[i]:
                new_position = best_solutions[0] + A * (np.random.rand(dim) - 0.5)
                new_fitness = obj_func(new_position)
                if new_fitness < fitness[i] and np.random.rand() < bats_loudness[i]:
                    bats_position[i] = new_position
                    fitness[i] = new_fitness
                    bats_loudness[i] *= alpha
                    bats_pulse_rate[i] = r_0 * (1 - np.exp(-gamma * iter))

        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < np.min(best_fitness):
            new_best = bats_position[min_fitness_idx]
            if is_new_solution(new_best, best_solutions):
                best_solutions.append(new_best)
                best_fitness.append(fitness[min_fitness_idx])

        if iter % 10 == 0:
            ax.cla()
            ax.scatter(bats_position[:, 0], bats_position[:, 1], bats_position[:, 2], color='blue', label='Bats', alpha=0.6)
            for sol in best_solutions:
                ax.scatter(sol[0], sol[1], sol[2], color='red', s=100, marker='X', label='Best Solution')
            ax.set_xlim(x_bounds)
            ax.set_ylim(x_bounds)
            ax.set_zlim(x_bounds)
            ax.set_title(f"Iteration {iter+1}, Best Fitness: {np.min(best_fitness):.3f}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.pause(0.1)

    plt.show()
    return best_solutions, best_fitness

# Tham số thuật toán
num_bats = 50
max_iter = 100
dim = 3
x_bounds = (-5.12, 5.12)
f_min = 0
f_max = 2
A = 0.5
r_0 = 1

# Chạy thuật toán
best_solutions, best_fitness = bat_algorithm_3d_multiple_solutions(
    objective_function, num_bats, max_iter, dim, x_bounds, f_min, f_max, A, r_0
)

# Kết quả
print("Best Solutions Found:")
for sol, fitness in zip(best_solutions, best_fitness):
    print(f"Solution: {sol}, Fitness: {fitness:.4f}")

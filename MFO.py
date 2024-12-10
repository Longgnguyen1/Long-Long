import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hàm mục tiêu (ví dụ: hàm Rastrigin trong không gian 3D)
def objective_function(x, y, z):
    A = 10
    return A * 3 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y)) + (z**2 - A * np.cos(2 * np.pi * z))

# Hàm kiểm tra tính đa dạng của các ngọn lửa
def is_new_flame(candidate, flames, threshold=0.5):
    for flame in flames:
        if np.linalg.norm(candidate - flame) < threshold:
            return False
    return True

# Thuật toán MFO với lưu nhiều điểm best flames
def moth_flame_optimization(obj_func, num_moths, max_iter, x_bounds, y_bounds, z_bounds, delay_time=0.1):
    # Khởi tạo ngẫu nhiên vị trí của các Moths
    moths = np.random.rand(num_moths, 3) * np.array([x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0], z_bounds[1] - z_bounds[0]]) + np.array([x_bounds[0], y_bounds[0], z_bounds[0]])

    # Lưu nhiều ngọn lửa tốt nhất (vị trí và giá trị)
    best_flames = []
    best_fitness = []

    # Tính giá trị hàm mục tiêu ban đầu
    fitness = np.apply_along_axis(lambda x: obj_func(x[0], x[1], x[2]), 1, moths)

    # Tạo đồ thị 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(x_bounds[0], x_bounds[1], 100)
    Y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = objective_function(X, Y, 0)  # Hàm tại mặt phẳng z=0

    plt.ion()  # Bật chế độ vẽ động

    for iter in range(max_iter):
        # Cập nhật ngọn lửa tốt nhất
        for i in range(num_moths):
            if is_new_flame(moths[i], best_flames, threshold=1.0):  # Ngưỡng phân biệt giữa các flame
                best_flames.append(moths[i].copy())
                best_fitness.append(fitness[i])

        # Chọn các flame tốt nhất (lọc theo fitness)
        sorted_indices = np.argsort(best_fitness)
        best_flames = [best_flames[i] for i in sorted_indices[:10]]  # Lấy tối đa 10 ngọn lửa tốt nhất
        best_fitness = [best_fitness[i] for i in sorted_indices[:10]]

        # Cập nhật vị trí của các Moths
        for i in range(num_moths):
            # Tính khoảng cách và di chuyển dần về flame gần nhất
            closest_flame = min(best_flames, key=lambda f: np.linalg.norm(moths[i] - f))
            distance_to_flame = closest_flame - moths[i]
            new_position = moths[i] + 0.05 * distance_to_flame  # Tốc độ di chuyển
            moths[i] = np.clip(new_position, [x_bounds[0], y_bounds[0], z_bounds[0]], [x_bounds[1], y_bounds[1], z_bounds[1]])

        # Cập nhật giá trị hàm mục tiêu
        fitness = np.apply_along_axis(lambda x: obj_func(x[0], x[1], x[2]), 1, moths)

        # Vẽ đồ thị mỗi vài vòng lặp
        if iter % 10 == 0:
            ax.clear()
            ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.6)
            ax.scatter(moths[:, 0], moths[:, 1], moths[:, 2], color='blue', label='Moths', alpha=0.6)
            for flame in best_flames:
                ax.scatter(flame[0], flame[1], flame[2], color='red', s=100, marker='X', label='Best Flame')
            ax.set_title(f"MFO Iteration {iter + 1}", fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc="upper right")
            plt.draw()
            plt.pause(delay_time)

    plt.ioff()
    plt.show()
    return best_flames, best_fitness

# Tham số thuật toán
num_moths = 25
max_iter = 100
x_bounds = (-6, 6)
y_bounds = (-6, 6)
z_bounds = (-6, 6)
delay_time = 0.5

# Chạy thuật toán
best_flames, best_fitness = moth_flame_optimization(objective_function, num_moths, max_iter, x_bounds, y_bounds, z_bounds, delay_time)

# Kết quả
print("Best Flames Found:")
for flame, fitness in zip(best_flames, best_fitness):
    print(f"Position: {flame}, Fitness: {fitness:.4f}")

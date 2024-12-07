import numpy as np

def gradient(w, x, y, z):
    gradienti_w = 2 * w - x - z + 3
    gradienti_x = 4 * x - w + 4
    gradienti_y = 6 * y + z - 5
    gradienti_z = 8 * z - w + y
    return np.array([gradienti_w, gradienti_x, gradienti_y, gradienti_z])

def function(w, x, y, z):
    return w**2 + 2 * x**2 + 3 * y**2 + 4 * z**2 - w * x + y * z - w * z + 3 * w + 4 * x - 5 * y + 6

data = [
    (0, 0, 0, 1),
    (1, -1, 2, -0.5),
    (1, -1, -1, 0.5)
]

learning_rate = 0.01
batch_size = 3
# itarations will be just 3, but I think more iteartions will give us the better resulsts

params = np.array([0.0, 0.0, 0.0, 1.0])


for i in range(3):
    grad_sum = np.zeros(4)
    for point in data:
        grad_sum += gradient(*point)
    avg_grad = grad_sum / batch_size

    
    params -= learning_rate * avg_grad

    func_value = function(*params)

    # here ill be printing the results
    print(f"Iteration {i + 1}:")
    print(f"Parameters: w = {params[0]:.8f}, x = {params[1]:.8f}, y = {params[2]:.8f}, z = {params[3]:.8f}")
    print(f"Function value: {func_value:.8f}\n")

import matplotlib.pyplot as plt
import numpy as np

# def cubic_bezier(t, p0, p1, p2, p3):
#     return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def cubic_bezier(t, p0, p1, p2, p3):
    return (
        (1 - t) ** 3 * np.array(p0)
        + 3 * (1 - t) ** 2 * t * np.array(p1)
        + 3 * (1 - t) * t ** 2 * np.array(p2)
        + t ** 3 * np.array(p3)
    )

def plot_cubic_bezier(p0, p1, p2, p3, num_points=100):
    t_values = np.linspace(0, 1, num_points)
    points = [cubic_bezier(t, p0, p1, p2, p3) for t in t_values]  # Calculate points separately
    x_values, y_values = zip(*points)
    plt.plot(x_values, y_values)
    plt.scatter([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Bezier Curve')
    plt.grid(True)
    plt.show()

plot_cubic_bezier((1, 1), (1.1, 6), (3.75, -1), (3, 10)) # just testing
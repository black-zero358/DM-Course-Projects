import matplotlib.pyplot as plt
import numpy as np

# Given data points
data = [
    [np.array([1.01051282, 0.01958974]), np.array([-0.83733333,  0.37066667]), np.array([ 0.04027322, -0.31634973])],
    [np.array([1.01051282, 0.01958974]), np.array([-0.83733333,  0.37066667]), np.array([ 0.04027322, -0.31634973])],
    [np.array([-0.83733333,  0.37066667]), np.array([1.01051282, 0.01958974]), np.array([ 0.04027322, -0.31634973])],
    [np.array([-0.83733333,  0.37066667]), np.array([1.01051282, 0.01958974]), np.array([ 0.04027322, -0.31634973])],
    [np.array([ 0.04027322, -0.31634973]), np.array([-0.83733333,  0.37066667]), np.array([1.01051282, 0.01958974])],
    [np.array([5.9016129, 2.7483871]), np.array([5.006, 3.428]), np.array([6.85, 3.07368421])]
]

# Colors for different points
colors = ['r', 'g', 'b']

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot each point movement and annotate each step
for i in range(3):
    x = [state[i][0] for state in data]
    y = [state[i][1] for state in data]
    plt.plot(x, y, marker='o', color=colors[i], label=f'Point {i + 1}')

    # Annotate each point with the step number
    for j in range(len(x)):
        plt.annotate(f'Step {j + 1}', (x[j], y[j]), textcoords="offset points", xytext=(0, -10), ha='center')

# Labels and title
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Movement of Points with Step Annotations')
plt.legend()
plt.grid(True)
plt.show()

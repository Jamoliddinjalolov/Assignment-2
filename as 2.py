import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2 - np.log(x)

def df(x):
    return 2*x - 1/x

# Initial point and learning rate
x0 = 2
alpha = 0.3

# Perform gradient descent steps
x1 = x0 - alpha * df(x0)
x2 = x1 - alpha * df(x1)

# Generate x values for function plotting
x_vals = np.linspace(-2.5, 2.5, 400)
y_vals = f(x_vals)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='f(x) = $x^2 - \log(x)$', color='black')

# Plot gradient descent points with distinctive colors
plt.scatter([x0], [f(x0)], color='green', s=100, label='$x^{(0)}$', zorder=5)  # Initial point
plt.scatter([x1], [f(x1)], color='blue', s=100, label='$x^{(1)}$', zorder=5)   # After 1st step
plt.scatter([x2], [f(x2)], color='red', s=100, label='$x^{(2)}$', zorder=5)    # After 2nd step

# Plot tangent lines at x0 and x1 with distinct styles
slope_x0 = df(x0)
intercept_x0 = f(x0) - slope_x0 * x0
tangent_line_x0 = slope_x0 * x_vals + intercept_x0
plt.plot(x_vals, tangent_line_x0, 'b--', label=f'Tangent at $x^{(0)}$', linewidth=2)

slope_x1 = df(x1)
intercept_x1 = f(x1) - slope_x1 * x1
tangent_line_x1 = slope_x1 * x_vals + intercept_x1
plt.plot(x_vals, tangent_line_x1, 'r--', label=f'Tangent at $x^{(1)}$', linewidth=2)

# Highlight arrows showing gradient descent steps with bold colors
plt.arrow(x0, f(x0), x1 - x0, f(x1) - f(x0), head_width=0.1, head_length=0.2, fc='blue', ec='blue', linewidth=2, zorder=4)
plt.arrow(x1, f(x1), x2 - x1, f(x2) - f(x1), head_width=0.1, head_length=0.2, fc='red', ec='red', linewidth=2, zorder=4)

# Add vertical dashed lines from points to x-axis with unique colors
plt.plot([x0, x0], [f(x0), 0], 'g--', linewidth=2)
plt.plot([x1, x1], [f(x1), 0], 'b--', linewidth=2)
plt.plot([x2, x2], [f(x2), 0], 'r--', linewidth=2)

# Customize plot appearance
plt.title('Gradient Descent Visualization', fontsize=16, fontweight='bold')
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.xticks([x0, x1, x2], [r"$x^{(0)}$", r"$x^{(1)}$", r"$x^{(2)}$"], fontsize=12)
plt.ylim([-0.5, 6])
plt.xlim([-0.5, 3])
plt.grid(True)

# Add legend and show plot
plt.legend(fontsize=12, loc='upper right')
plt.show()

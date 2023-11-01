# import numpy as np
# import matplotlib.pyplot as plt

# # Define the function
# def polarizing_amplified_leaky_relu(x):
#     if x <= -1:
#         return -100 * x
#     elif -1 < x < 1:
#         return 0
#     else:  # x >= 1
#         return x

# # Generate a range of x values
# x_values = np.linspace(-2.55, 2.55, 400)

# # Apply the custom activation function to each x value
# y_values = [polarizing_amplified_leaky_relu(x) for x in x_values]

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, y_values, label="Image Pixel Distribution Polarizing Amplified Leaky ReLU")
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.title("Image Pixel Distribution Polarizing Amplified Leaky ReLU")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.legend()

# # Save the plot
# save_dir = '/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir/actiation_funcs_plots/'
# plt.savefig(save_dir+ "Image Pixel Distribution Polarizing Amplified Leaky ReLU.png")

# plt.show()

#--------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Define the activation function
# def polarized_amplified_leaky_relu(x):
#     if x <= -1:
#         return -100
#     elif -1 < x < 1:
#         return 0
#     else:  # x >= 1
#         return 1

# # Define the function for applying the activation
# def apply_activation(x):
#     if x <= -1:
#         return -100 * x
#     elif -1 < x < 1:
#         return 0
#     else:  # x >= 1
#         return x

# # Generate a normal distribution of x values within the range [-2.5, 2.5]
# x_vals = np.random.normal(loc=0, scale=1, size=10000)
# x_vals = x_vals[(x_vals >= -2.5) & (x_vals <= 2.5)]
# x_vals_sorted = np.sort(x_vals)

# # Create a figure with three subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# # Plot the normal distribution curve
# density_curve_x = np.linspace(-2.5, 2.5, 1000)
# density_curve_y = norm.pdf(density_curve_x, loc=0, scale=1)
# axs[0].plot(density_curve_x, density_curve_y, color='blue')
# axs[0].set_title("Distribution of Images after Normalization with mean and std")
# axs[0].set_xlabel("x")
# axs[0].set_ylabel("Density")
# axs[0].grid(True)

# # Plot the activation function itself
# x_plot = np.linspace(-2.5, 2.5, 1000)
# y_plot = [polarized_amplified_leaky_relu(x) for x in x_plot]
# axs[1].step(x_plot, y_plot, where='post')
# axs[1].set_title("Polarized Amplified Leaky ReLU Activation Function")
# axs[1].set_xlabel("x")
# axs[1].set_ylabel("y")
# axs[1].grid(True)

# # Generate activated values
# activated_values = [apply_activation(x) for x in x_vals_sorted]

# # Plot the x_vals after activation using a line plot with thicker line width
# axs[2].plot(x_vals_sorted, activated_values, color='blue', linewidth=2)
# axs[2].set_title("Images After Activation Fucntion")
# axs[2].set_xlabel("x")
# axs[2].set_ylabel("y")
# axs[2].grid(True)

# # Adjust the layout
# plt.tight_layout()

# # Save the combined plot
# save_dir = '/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir/actiation_funcs_plots/'
# plt.savefig(save_dir + "Polarized_Amplified_Leaky_ReLU_Three_Plots.png")

# plt.show()


### --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the original activation function
def polarized_amplified_leaky_relu(x):
    if x <= -1:
        return -100
    elif -1 < x < 1:
        return 0
    else:  # x >= 1
        return 1

# Define the function for applying the original activation
def apply_activation(x):
    if x <= -1:
        return -100 * x
    elif -1 < x < 1:
        return 0
    else:  # x >= 1
        return x

# Define the function for applying the modified activation
def apply_modified_activation(x):
    if x <= -1:
        return -100 * (x + 1)
    elif -1 < x < 1:
        return 0
    else:  # x >= 1
        return x

# Generate a normal distribution of x values within the range [-2.5, 2.5]
x_vals = np.random.normal(loc=0, scale=1, size=10000)
x_vals = x_vals[(x_vals >= -2.5) & (x_vals <= 2.5)]
x_vals_sorted = np.sort(x_vals)

# Create a figure with six subplots (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot the normal distribution curve in both rows
density_curve_x = np.linspace(-2.5, 2.5, 1000)
density_curve_y = norm.pdf(density_curve_x, loc=0, scale=1)
for row in range(2):
    axs[row, 0].plot(density_curve_x, density_curve_y, color='blue')
    axs[row, 0].set_title("Normal Distribution of Images")
    axs[row, 0].set_xlabel("x")
    axs[row, 0].set_ylabel("Density")
    axs[row, 0].grid(True)

# Plot the original activation function
x_plot = np.linspace(-2.5, 2.5, 1000)
y_plot = [polarized_amplified_leaky_relu(x) for x in x_plot]
axs[0, 1].step(x_plot, y_plot, where='post')
axs[0, 1].set_title("Polarized Amplified Leaky ReLU Activation Function")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
axs[0, 1].grid(True)

# Plot the modified activation function in the second row
axs[1, 1].step(x_plot, y_plot, where='post')  # The function shape remains the same
axs[1, 1].set_title("Modified Polarized Amplified Leaky ReLU Activation Function")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
axs[1, 1].grid(True)

# Generate activated values using the original activation
activated_values = [apply_activation(x) for x in x_vals_sorted]
axs[0, 2].plot(x_vals_sorted, activated_values, color='blue', linewidth=2)
axs[0, 2].set_title("Images After Original Activation")
axs[0, 2].set_xlabel("x")
axs[0, 2].set_ylabel("y")
axs[0, 2].grid(True)

# Generate activated values using the modified activation
activated_modified_values = [apply_modified_activation(x) for x in x_vals_sorted]
axs[1, 2].plot(x_vals_sorted, activated_modified_values, color='blue', linewidth=2)
axs[1, 2].set_title("Images After Modified Activation")
axs[1, 2].set_xlabel("x")
axs[1, 2].set_ylabel("y")
axs[1, 2].grid(True)

# Adjust the layout
plt.tight_layout()

# Save the combined plot
save_dir = '/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir/actiation_funcs_plots/'
plt.savefig(save_dir + "Polarized_Amplified_Leaky_ReLU_Six_Plots.png")

plt.show()

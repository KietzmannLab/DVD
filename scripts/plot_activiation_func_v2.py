import numpy as np
import matplotlib.pyplot as plt

def custom_steep_double_sigmoid(x, x0=-1, k=0.05):

    # Apply the condition to each element of the array
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k)), 
                 -2 / (1 + np.exp((-x - x0) / k)))
    return y

def custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05):

    # Apply the condition to each element of the array
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k)), 
                 2 / (1 + np.exp((-x - x0) / k)))
    return y

def custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=0, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

# Generate a range of values from -10 to 10, which will include our sigmoid changes
x_values = np.linspace(-2.55, 2.55, 1000)
# y_values = custom_steep_positive_double_sigmoid(x_values)

# plt.figure(figsize=(8, 6))
# plt.plot(x_values, y_values, label=f"Custom Sigmoid, x0={-1/1}, k={0.05}")
# plt.title("Custom Steep Sigmoid Function")
# plt.xlabel("x")
# plt.ylabel("Sigmoid Output")
# plt.legend()
# plt.grid(True)

# # Save the plot as a file
# plt.savefig('save_dir/actiation_funcs_plots/custom_steep_positive_double_sigmoid.png', format='png')

# # Show the plot
# plt.show()
# Generate the y-values for each function
y_values_double_sigmoid = custom_steep_double_sigmoid(x_values)
y_values_positive_double_sigmoid = custom_steep_positive_double_sigmoid(x_values)
y_values_shifted_minus_1_steep = custom_shifted_minus_1_steep_sigmoid(x_values)
y_values_shifted_minus_1_not_steep = custom_shifted_minus_1_not_steep_sigmoid(x_values)
y_values_shifted_steep = custom_shifted_steep_sigmoid(x_values)
y_values_shifted_not_steep = custom_shifted_not_steep_sigmoid(x_values)

# Plot each sigmoid function with a unique label and color
plt.plot(x_values, y_values_double_sigmoid, label='Steep Double Sigmoid', color='blue')
plt.plot(x_values, y_values_positive_double_sigmoid, label='Steep Positive Double Sigmoid', color='red')
plt.plot(x_values, y_values_shifted_minus_1_steep, label='Shifted -1 Steep Sigmoid', color='green')
plt.plot(x_values, y_values_shifted_minus_1_not_steep, label='Shifted -1 Not Steep Sigmoid', color='purple')
plt.plot(x_values, y_values_shifted_steep, label='Shifted Steep Sigmoid', color='orange')
plt.plot(x_values, y_values_shifted_not_steep, label='Shifted Not Steep Sigmoid', color='black')

# Add title and labels
plt.title("Comparison of Custom Sigmoid Functions")
plt.xlabel("x")
plt.ylabel("Sigmoid Output")
plt.legend()
plt.grid(True)

# Save the plot as a file
plt.savefig('save_dir/actiation_funcs_plots/comparison_custom_sigmoids.png', format='png')

# Show the plot
plt.show()
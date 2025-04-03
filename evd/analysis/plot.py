import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Set the style for the plots
plt.style.use('nature')
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 5

def plot_shape_bias_across_time(csv_path, colors = ['orange', 'black', 'blue', 'green', 'red'], save_path=None):
    """
    Reads shape-bias data from a CSV file and plots how shape bias evolves over time.

    CSV is assumed to have columns:
      model_name,epoch,top1,top5,shape_bias,timepoint

    The 'shape_bias' column is in the form '[0.5]', so we parse out the float value.
    The time in months is computed as (epoch + 1) * 2.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Parse shape_bias from the bracketed string "[x]" into a float
    df['shape_bias'] = df['shape_bias'].str.strip('[]').astype(float)

    # Compute time in months from epoch
    df['months'] = (df['epoch'] + 1) * 2

    # Create the plot
    plt.figure(figsize=(3.54, 2))

    # Plot each model's shape bias over time
    for model_name in df['model_name'].unique():
        subset = df[df['model_name'] == model_name]
        plt.plot(subset['months'], subset['shape_bias'], marker='o', label=model_name, color=colors.pop(0))

    # Labeling and styling
    plt.xlabel('Time (Months)')
    plt.ylabel('Shape Bias')
    plt.title('Shape Bias Development Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)



if __name__ == "__main__":
    # Example usage
    csv_file_path = '/home/student/l/lzejin/codebase/All-TNNs/P001_evd_gpus/results/shape_bias/19th_Wed_traverse_4_shape_bais_plot_1.csv'
    plot_shape_bias_across_time(csv_file_path, save_path='/home/student/l/lzejin/codebase/All-TNNs/P001_evd_gpus/results/plots/shape_bias/shape_bias_plot_B2.pdf')
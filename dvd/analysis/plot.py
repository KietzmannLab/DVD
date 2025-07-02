import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from scipy.io import loadmat
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import ast
from io import StringIO

from neuroai.plotting.colors import *

# Set the style for the plots
plt.style.use('nature')
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5


def plot_shape_bias_across_time(csv_path, colors=['steelblue', 'black', 'blue', 'orange', 'red'], include_humans=True, save_path=None):
    """
    Reads shape-bias data from a CSV file and plots how shape bias evolves over time.
    Optionally includes human developmental data points.

    CSV is assumed to have columns:
      model_name,epoch,top1,top5,shape_bias,timepoint

    The 'shape_bias' column is a float.
    The time in months is computed as (epoch + 1) * 2.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Ensure shape_bias is a float
    df['shape_bias'] = df['shape_bias'].astype(float)

    # Compute time in months from epoch
    df['months'] = (df['epoch'] + 1) * 2

    # Create the plot
    plt.figure(figsize=(3.54, 2))

    # Plot each model's shape bias over time
    for model_name in df['model_name'].unique():
        subset = df[df['model_name'] == model_name]
        plt.plot(subset['months'], subset['shape_bias'], marker='o', label=model_name, color=colors.pop(0))

    # Optionally add human developmental data
    if include_humans:
        human_data = [
            ("humans(4-6 y/o)", (0.8116, 0.5411, 0.6), 0.8755274262, 60),
            ("humans(7-9 y/o)", (0.8117, 0.5411, 0.6), 0.9085714286, 96),
            ("humans(10-12 y/o)", (0.8117, 0.5411, 0.6), 0.9352226721, 132),
            ("humans(13-15 y/o)", (0.8117, 0.5411, 0.6), 0.9318996416, 168),
            ("humans(adult)", (0.6471, 0.1176, 0.2157), 0.9586879801, 282),
        ]
        for label, color, shape_bias, month in human_data:
            plt.plot(month, shape_bias, marker='o', markersize=5, label=label, color=color)

    # Labeling and styling
    plt.xlabel('Time (Months)')
    plt.ylabel('Shape Bias')
    plt.title('Shape Bias Development Over Time')
    plt.legend(fontsize='xx-small', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_shape_and_texture_accuracy_across_time(csv_path, colors=[np.array([49, 162, 142]) / 255, 'black'], save_path=None):
    """
    Plots shape accuracy and texture accuracy of models across time using data from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        colors (list): List of colors for the plot lines [shape_color, texture_color].
        save_path (str): If specified, saves the plot to this path.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Convert 'shape_acc' and 'texture_acc' from strings of lists to actual lists of floats
    df['shape_acc'] = df['shape_acc'].apply(eval)
    df['texture_acc'] = df['texture_acc'].apply(eval)

    # Compute average shape and texture accuracy across classes for each row
    df['shape_acc_mean'] = df['shape_acc'].apply(lambda x: sum(x) / len(x))
    df['texture_acc_mean'] = df['texture_acc'].apply(lambda x: sum(x) / len(x))

    # Compute time in months from epoch
    df['months'] = (df['epoch'] + 1) * 2

    # Create the plot
    plt.figure(figsize=(3.54, 2))

    markers = ['o', 's', '^', 'D', 'v', 'P']  # variety of shapes
    for idx, model_name in enumerate(df['model_name'].unique()):
        subset = df[df['model_name'] == model_name]
        # m = markers #[idx % len(markers)]
        plt.plot(subset['months'], subset['shape_acc_mean'], label=f'{model_name} - Shape', color=colors[idx], linestyle='-',
                 ) # marker=m) #, markersize=1, markeredgewidth=1, )
        plt.plot(subset['months'], subset['texture_acc_mean'], label=f'{model_name} - Texture', color=colors[idx], linestyle='--',
                 ) # marker=m) #, markersize=1, markeredgewidth=1, )
        
    plt.xlabel('Time (Months)')
    plt.ylabel('Accuracy')
    plt.title('Shape and Texture Accuracy Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def bar_plot(values, tick_labels, x_label, y_label, title, save_path, figsize=(3.54, 2)):
    """
    Creates and saves a bar plot.
    
    Args:
        values (list or np.array): The heights of the bars.
        tick_labels (list): Labels for the x-ticks.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        save_path (str): File path (without extension) to save the plot.
        figsize (tuple): Figure size in inches.
    """
    plt.figure(figsize=figsize)
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, tick_labels, rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close()


def curve_plot(values, tick_labels, x_label='x', y_label='y', title=None, save_path=None, figsize=(3.54, 2)):
    """
    Creates and saves a curve (line) plot.
    
    Args:
        values (list or np.array): y-values of the curve.
        tick_labels (list): Labels for the x-ticks.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str, optional): Title of the plot.
        save_path (str, optional): File path (without extension) to save the plot.
        figsize (tuple): Figure size in inches.
    """
    plt.figure(figsize=figsize)
    x = np.arange(len(values))
    plt.plot(x, values, marker='o', linestyle='-')
    plt.xticks(x, tick_labels, rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close()


def multi_curve_plot(pivot_df, x_values, x_label, y_label, title, save_path, model_colors, figsize=(3.54/2, 2)):
    """
    Creates and saves a multi-line curve plot using real x values.
    
    Args:
        pivot_df (DataFrame): A DataFrame with epsilon values as the index and models as columns.
        x_values (list): List of actual epsilon values (used as x coordinates).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Plot title.
        save_path (str): Full file path to save the plot.
        model_colors (dict): Mapping from model name to a matplotlib-compatible color.
        figsize (tuple): Figure size in inches.
    """
    plt.figure(figsize=figsize)
    
    # Plot each model's curve using the actual epsilon values on the x-axis.
    for model in pivot_df.columns:
        plt.plot(x_values, pivot_df[model].values, marker='o', linestyle='-', label=model,
                 color=model_colors.get(model))
    
    # Set x-ticks to the actual epsilon values and label them appropriately.
    plt.xticks(x_values, [int(x) for x in x_values])#, rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        print(f'Saving figure at {save_path}')
        plt.savefig(save_path, dpi=300)
        plt.close()



def multi_curve_on_ax(ax, pivot_df, x_values, x_label, y_label, title,
                      model_colors, show_legend=False, font_size=5):
    for model in pivot_df.columns:
        ax.plot(x_values, pivot_df[model].values,
                marker='o', linestyle='-',
                label=model, color=model_colors.get(model))

    ax.set_xticks(x_values)
    ax.set_xticklabels([str(int(x)) for x in x_values], fontsize=font_size)

    # Always show y-tick labels; hide axis label only if not first col
    if y_label:
        ax.set_ylabel('Accuracy', fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

    if x_label:
        ax.set_xlabel('Severity', fontsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)

    # Title closer to plot
    ax.set_title(title, fontsize=font_size, y=0.94)

    # Light, thin grid lines
    ax.grid(True, which='major', linestyle='-', linewidth=0.3, alpha=0.2)

    if show_legend:
        ax.legend(fontsize=font_size)

def plot_degradation_grid(degradation_df, save_dir,
                          figsize=(7, 8),
                          save_name="degradation_robustness_grid.pdf",
                          model_colors=None,
                          show_legend=False,
                          relative=False):
    """
    Plots a 4×4 grid of severity curves for each distortion type,
    using clean accuracy as severity 0 for each.
    
    If `relative` is True, accuracy is normalized to clean accuracy per model.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Extract clean values
    clean_df = degradation_df[
        (degradation_df['distortion'] == 'clean') &
        (degradation_df['severity'] == 0)
    ][['model', 'accuracy']].set_index('model')['accuracy']

    # Preserve original distortion order from CSV
    baseline_rows = degradation_df[degradation_df['model'] == degradation_df['model'].unique()[0]]
    distortion_order = baseline_rows[
        (baseline_rows['distortion'] != 'clean')
    ]['distortion'].drop_duplicates().tolist()

    models = sorted(degradation_df['model'].unique())

    if model_colors is None:
        default_colors = [teal_green, black, bright_green, deep_blue, 'red', 'purple']
        model_colors = {m: default_colors[i % len(default_colors)] for i, m in enumerate(models)}

    n = len(distortion_order)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for idx, (ax, dtype) in enumerate(zip(axes, distortion_order)):
        sub = degradation_df[degradation_df['distortion'] == dtype]
        pivot = sub.pivot(index='severity', columns='model', values='accuracy')
        pivot = pivot.sort_index()

        # Add clean as severity 0
        pivot.loc[0] = clean_df
        pivot = pivot.sort_index()

        if relative:
            pivot = pivot.divide(clean_df, axis=1)

        x_vals = pivot.index.tolist()

        col_idx = idx % ncols
        row_idx = idx // ncols
        show_y = col_idx == 0
        show_x = row_idx == nrows - 1

        multi_curve_on_ax(ax, pivot, x_vals,
                          x_label=show_x,
                          y_label=show_y,
                          title=dtype,
                          model_colors=model_colors,
                          show_legend=show_legend,
                          font_size=5)

    for ax in axes[len(distortion_order):]:
        fig.delaxes(ax)

    plt.tight_layout(pad=0.5)
    save_path = os.path.join(save_dir, save_name)
    print(f"Saving full grid to {save_path}")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_adv_attack_results_comparison_multi(attack_list, results_df, save_dir, model_colors=None, figsize=(3.54 * 0.55, 2 * 0.85)):
    """
    Generates a row of subplots comparing accuracy curves for multiple adversarial attacks,
    where x positions are proportional to real epsilon values but labeled by index.

    Args:
        attack_list (list): List of attack names.
        results_df (DataFrame): DataFrame with columns: Model, Attack, Epsilon, Accuracy, Epoch.
        save_dir (str): Directory where the plot will be saved.
        model_colors (dict, optional): Mapping of models to colors.
        figsize_per_plot (tuple): Size of each individual subplot.
    """
    n_attacks = len(attack_list)

    fig, axes = plt.subplots(1, n_attacks, figsize=figsize)

    if n_attacks == 1:
        axes = [axes]

    for ax, attack in zip(axes, attack_list):
        attack_df = results_df[results_df['Attack'] == attack]
        if attack_df.empty:
            print(f"No results found for attack: {attack}")
            continue

        pivot_df = attack_df.pivot(index='Epsilon', columns='Model', values='Accuracy').sort_index()
        epsilons = pivot_df.index.tolist()
        models = pivot_df.columns.tolist()

        if model_colors is None:
            default_colors = [black, teal_green, bright_green, deep_blue, 'red', 'purple']  #['black', 'teal', 'green', 'blue', 'red', 'purple']
            model_colors = {model: default_colors[i % len(default_colors)] for i, model in enumerate(models)}

        for model in models:
            ax.plot(epsilons, pivot_df[model].values, marker='o', linestyle='-',
                    label=model if ax == axes[0] else "",
                    color=model_colors.get(model))

        # Set proportional x position but index labels
        ax.set_xticks(epsilons)
        ax.set_xticklabels(range(len(epsilons)))
        ax.set_xlabel('epsilon strength')
        ax.set_title(attack)

    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "accuracy_comparison_all_attacks_curve_indices.pdf")
    print(f"Saving figure at {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_adv_attack_results_comparison(attack, results_df, save_dir, model_colors=None, figsize=(3.54*0.55, 2*0.85)):
    """
    Filters the results DataFrame for a given adversarial attack,
    pivots the data to compare the accuracy of different models under real epsilon values,
    and generates a multi-line curve plot where the x-axis reflects the actual epsilon distances.
    
    Args:
        attack (str): The attack name to filter (e.g., 'L2AdditiveGaussianNoiseAttack').
        results_df (DataFrame): DataFrame containing columns: Model, Attack, Epsilon, Accuracy, Epoch.
        save_dir (str): Directory path where the plot will be saved.
        model_colors (dict, optional): Optional mapping of models to colors (e.g., {'baseline': 'gray', 'baseline + TVD': 'green'}).
        figsize (tuple): Figure size for the plot.
    """
    # Filter the DataFrame for the specified attack.
    attack_df = results_df[results_df['Attack'] == attack]
    if attack_df.empty:
        print(f"No results found for attack: {attack}")
        return
    
    # Pivot the DataFrame so that rows represent real epsilon values and columns represent models.
    pivot_df = attack_df.pivot(index='Epsilon', columns='Model', values='Accuracy')
    pivot_df = pivot_df.sort_index()  # Ensure epsilon values are sorted numerically.
    x_values = pivot_df.index.tolist()  # These are the actual epsilon values.
    
    # Determine colors for models if not provided.
    models = pivot_df.columns.tolist()
    if model_colors is None:
        if len(models) == 2:
            model_colors = {models[0]: black, models[1]: teal_green}
        else:
            default_colors = [black, teal_green, bright_green, deep_blue, 'red', 'purple'] #['lightgray', 'green', 'blue', 'orange', 'red', 'purple']
            model_colors = {model: default_colors[i % len(default_colors)] for i, model in enumerate(models)}
    
    # Create the plot title.
    plot_title = f'{attack}' # Accuracy Comparison under 
    
    # Ensure the save directory exists and define the path to save the plot.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"accuracy_comparison_{attack}_curve.pdf")
    
    # Create and save the multi-model curve plot.
    multi_curve_plot(pivot_df, x_values, 'Epsilon (Relative Distance)', 'Accuracy', plot_title, save_path, model_colors, figsize)


def plot_sweeped_shape_bias_tradeoff_scatter(csv_path: str,
                                       analysis_name: str,
                                       save_dir: str = "./results/plots/trade_off",
                                       linear_fit: bool = False,
                                       plot_settings: dict = None,
                                       xlim: tuple = (40, 75),
                                       ylim: tuple = (0.3, 1),
                                       annotate: bool = False,
                                       degree: int = 3,
                                       figsize: tuple = (3.5*0.66, 2),
                                       topk: int = 1,
                                       ):
    """
    Reads a CSV of model results, parses shape_bias, splits into 'adult' vs. others,
    optionally fits a linear line or polynomial to non-adult points,
    annotates points if requested, and saves a PDF plot using styles from plot_settings.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    analysis_name : str
        Base name for the output file (no extension).
    save_dir : str
        Directory to save the PDF into.
    linear_fit : bool
        If True, fit and plot a linear model y = w x + b (light gray, equation shown).
    annotate : bool
        If True, annotate each point with its (mpe, alpha, dn) or 'adult'.
    degree : int
        Degree of polynomial fit (used only if linear_fit is False).
    plot_settings : dict
        Mapping categories to style dicts, e.g.:
        {
          'baseline': {'color':'black','marker':'o','size':3},
          'TVD-S'   : {'color':'blue',  'marker':'*','size':3},
          'TVD-B'   : {'color':'green', 'marker':'*','size':3},
          'TVD-P'   : {'color':'purple','marker':'*','size':3},
          'TVD-PP'  : {'color':'red',   'marker':'*','size':3},
        }
    """
    # Default styles
    if plot_settings is None:
        plot_settings = {
            'baseline': {'color':'black','marker':'o','size':4},
            'TVD-S'   : {'color':'blue',  'marker':'*','size':4},
            'TVD-B'   : {'color':'green', 'marker':'*','size':4},
            'TVD-P'   : {'color':'purple','marker':'*','size':4},
            'TVD-PP'  : {'color':'red',   'marker':'*','size':4},
        }

    def parse_model_name(name: str):
        parts = name.split('_')
        try:
            mpe   = float(parts[0].replace('mpe', ''))
            alpha = float(parts[1].replace('alpha', ''))
            dn    = float(parts[2].replace('dn', ''))
            return mpe, alpha, dn
        except:
            return None, None, None

    mpe_to_cat = {1.0: 'TVD-S', 2.0: 'TVD-B', 4.0: 'TVD-P', 8.0: 'TVD-PP'}

    # Prepare
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    # df['shape_bias'] = df['shape_bias'].astype(str).str.strip('[]').astype(float)
    df['shape_bias'] = df['shape_bias'].astype(str).str.strip('[]').astype(float)

    df_adult    = df[df['model_name'] == 'adult']
    df_nonadult = df[df['model_name'] != 'adult']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot baseline
    bs = plot_settings['baseline']
    if not df_adult.empty:
        ax.scatter(df_adult[f'top{topk}'], df_adult['shape_bias'],
                   s=bs['size'], c=bs['color'], marker=bs['marker'],
                   label='baseline')

    # Plot TVD categories
    for mpe_val, cat in mpe_to_cat.items():
        settings = plot_settings.get(cat)
        sub = df_nonadult[df_nonadult['model_name'].apply(lambda m: parse_model_name(m)[0] == mpe_val)]
        if settings and not sub.empty:
            ax.scatter(sub[f'top{topk}'], sub['shape_bias'],
                       s=settings['size'], c=settings['color'],
                       marker=settings['marker'], label=cat)

    # Optional annotation
    if annotate:
        for _, row in df_nonadult.iterrows():
            mpe, alpha, dn = parse_model_name(row['model_name'])
            if mpe is not None:
                ax.annotate(f"({mpe},{alpha},{dn})",
                            (row[f'top{topk}'], row['shape_bias']),
                            textcoords="offset points", xytext=(5, 5), fontsize=6)
        for _, row in df_adult.iterrows():
            ax.annotate("adult",
                        (row[f'top{topk}'], row['shape_bias']),
                        textcoords="offset points", xytext=(5, 5), fontsize=6)

    # Prepare fit x-range
    x_min, x_max = df_nonadult[f'top{topk}'].min(), df_nonadult[f'top{topk}'].max()
    x_fit = np.linspace(x_min, x_max, 100)

    # Linear fit
    if linear_fit:
        w, b = np.polyfit(df_nonadult[f'top{topk}'], df_nonadult['shape_bias'], deg=1)
        y_lin = w * x_fit + b
        ax.plot(x_fit, y_lin, '--', lw=1.0, color='lightgray', label=f'y = {w:.3f}x + {b:.3f}')
        # eq_text = f'y = {w:.3f}x + {b:.3f}'
        # ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
        #         verticalalignment='top', fontsize=8, color='gray')
    else:
        coeffs = np.polyfit(df_nonadult[f'top{topk}'], df_nonadult['shape_bias'], deg=degree)
        poly = np.poly1d(coeffs)
        ax.plot(x_fit, poly(x_fit), '-', lw=2, color='red', label=f'Poly(deg={degree})')

    # Labels & legend
    ax.set_xlabel(f"Accuracy (top1)", fontsize = 6)
    ax.set_ylabel("Shape Bias", fontsize = 6)
    ax.set_title("Accuracy - shape Bias tradeoff", fontsize = 7)

    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Legend
    ax.legend(fontsize=5, loc='best')
    plt.tight_layout()

    out_file = os.path.join(save_dir, f"{analysis_name}_sweeped_trade_off_top1.pdf")
    plt.savefig(out_file)
    plt.close(fig)

    print(f"✔ Saved plot to {out_file}")


def plot_shape_bias_tradeoff_scatter(csv_path: str,
                          analysis_name: str,
                          save_dir: str = "./results/plots/trade_off",
                          plot_settings: dict = None,
                          xlim: tuple = (40, 70),
                          ylim: tuple = (0.3, 1),
                          figsize: tuple = (3.5*0.66, 2),
                          topk: int = 1,
                          ):
    """
    Reads a CSV, cleans shape_bias, and creates a styled scatter plot per model category.
    Saves the figure as a PDF.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing columns: model_name, top1, shape_bias.
    analysis_name : str
        Base name for the output PDF (no extension).
    save_dir : str
        Directory to save the PDF into.
    plot_settings : dict, optional
        Mapping of model_name to style dicts
        If None, defaults will be used.
    xlim : tuple, optional
        X-axis limits (min, max).
    ylim : tuple, optional
        Y-axis limits (min, max).
    figsize : tuple, optional
        Figure size in inches.
    """
    # Default plot settings
    if plot_settings is None:
        plot_settings = {
            'baseline': {'color':'black', 'marker':'o', 'size':10},
            'TVD-P':    {'color':'purple','marker':'*','size':20},
            'TVD-PP':   {'color':'red',   'marker':'*','size':20},
            'TVD-B':    {'color':'green', 'marker':'*','size':20},
            'TVD-S':    {'color':'blue',  'marker':'*','size':20},
            'TVD-SS':    {'color':'blue',  'marker':'*','size':20},
        }

    # Prepare save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path, usecols=['model_name', f'top{topk}', 'shape_bias'])
    # Ensure shape_bias is float
    df['shape_bias'] = df['shape_bias'].astype(float)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each model group
    for model, style in plot_settings.items():
        subset = df[df['model_name'] == model]
        if subset.empty:
            continue
        ax.scatter(
            subset[f'top{topk}'], subset['shape_bias'],
            s=style['size'],
            c=style['color'],
            marker=style['marker'],
            label=model
        )

    # Labels and title
    ax.set_xlabel("Top1 Accuracy", fontsize=6)
    ax.set_ylabel("Shape Bias", fontsize=6)
    ax.set_title("Accuracy - shape Bias tradeoff", fontsize=7)

    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Legend
    ax.legend(fontsize=5, loc='lower left')

    plt.tight_layout()

    # Save output
    out_file = os.path.join(save_dir, f"{analysis_name}_trade_off.pdf")
    plt.savefig(out_file)
    plt.close(fig)

    print(f"✔ Saved plot to {out_file}")


#* mapping colormap by order

def plot_sweeped_shape_bias_tradeoff_scatter_colormap(
    csv_path: str,
    analysis_name: str,
    save_dir: str = "./results/plots/trade_off",
    highlight_models: list = None,
    linear_fit: bool = False,
    annotate: bool = False,
    degree: int = 3,
    figsize: tuple = (3.5 * 0.66, 2),
    topk: int = 1,
    cmap_name: str = "magma",
    reverse: bool = False,
    use_fixed_range: bool = False,
    fixed_range: tuple = (0.0, 1.0),
    xlim: tuple = (40, 75),
    ylim: tuple = (0.3, 1.0),
    plot_epoch = ['best', 'all'][1],
    print_model_colors: bool = True,  # New parameter
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if plot_epoch == 'best':
        df = df.loc[df['epoch'] == 'best' ] # if only plot best epoch
    df['shape_bias'] = df['shape_bias'].astype(str).str.strip('[]').astype(float)

    x_col = f"top{topk}"
    x_all = df[x_col].values
    y_all = df['shape_bias'].values
    names = df['model_name'].values

    # sort so high bias models draw on top
    order = np.argsort(-y_all)
    x = x_all[order]
    y = y_all[order]
    names = names[order]
    rows = df.iloc[order]

    if highlight_models is None:
        highlight_models = []

    is_baseline = np.array([('adult' in name or 'baseline' in name) for name in names])
    # is_highlight = np.array([name in highlight_models for name in names])
    is_highlight = np.array([
                            (name in highlight_models) and (epoch == 'best')
                            for name, epoch in zip(names, df['epoch'].values[order])
                        ])
    is_other = ~is_baseline & ~is_highlight

    cmap = mpl.cm.get_cmap(cmap_name + ("_r" if reverse else ""))
    norm = mpl.colors.Normalize(vmin=fixed_range[0], vmax=fixed_range[1]) if use_fixed_range else mpl.colors.Normalize(vmin=y.min(), vmax=y.max())

    fig, ax = plt.subplots(figsize=figsize)

    # 1) baseline: light gray
    scatter_base = ax.scatter(
        x[is_baseline], y[is_baseline],
        c='lightgray', s=10, edgecolor='k', lw=0, alpha=0.8,
        marker='o', label='Baseline'
    )

    # 2) highlighted: stars
    scatter_highlight = ax.scatter(
        x[is_highlight], y[is_highlight],
        c=cmap(norm(y[is_highlight])),
        s=50, edgecolor='k', lw=0.2, alpha=0.9,
        marker='*', label='Highlighted'
    )

    # 3) other models: circles
    ax.scatter(
        x[is_other], y[is_other],
        c=cmap(norm(y[is_other])),
        s=10, edgecolor='k', lw=0, alpha=0.8,
        marker='o'
    )

    # colorbar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, pad=0.03, fraction=0.08)
    cbar.set_label("Shape Bias", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # optional annotations
    if annotate:
        for xi, yi, name in zip(x, y, names):
            label = "adult" if name == "adult" else name
            ax.annotate(label, (xi, yi),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=4, color='black')

    # optional trendline (on non-adult models)
    mask = df['model_name'] != 'adult'
    x_fit = df.loc[mask, x_col].values
    y_fit = df.loc[mask, 'shape_bias'].values

    coeffs = np.polyfit(x_fit, y_fit, 1 if linear_fit else degree)
    label_fit = f"Linear: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}" if linear_fit else f"Poly (deg={degree})"
    xs = np.linspace(x_fit.min(), x_fit.max(), 200)
    ys = np.polyval(coeffs, xs)
    ax.plot(xs, ys, '--', color='gray', lw=1, label=label_fit)

    # axes setup
    ax.set_xlabel(f"Accuracy (top{topk})", fontsize=6)
    ax.set_ylabel("Shape Bias", fontsize=6)
    ax.set_title("Accuracy vs. Shape Bias", fontsize=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.tick_params(labelsize=5)

    # Add horizontal dashed lines for human benchmarks
    ax.axhline(y=0.96, color='green', linestyle='--', linewidth=1, label='Human (adult)')
    ax.axhline(y=0.88, color='green', linestyle='--', linewidth=1, label='Human (4–6 y/o)')

    # show only legend for baseline, highlighted, and fit line
    handles = [scatter_base, scatter_highlight, ax.lines[-1]]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, fontsize=5, framealpha=0.5)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{analysis_name}_sweeped_colormap_top{topk}.pdf")
    plt.savefig(out)
    plt.close(fig)
    print(f"✔ Saved plot to {out}")

    # Optional: Print model names and their RGB colors
    if print_model_colors:
        model_color_map = {}
        for name, bias in zip(names, y):
            rgba = cmap(norm(bias))
            rgb = tuple(round(c, 3) for c in rgba[:3])
            model_color_map[name] = rgb
        print("Model to RGB color mapping:")
        for name, rgb in model_color_map.items():
            print(f"{name}: {rgb}")

def plot_shape_bias_tradeoff_scatter_colormap(
    csv_path: str,
    analysis_name: str,
    save_dir: str = "./results/plots/trade_off",
    topk: int = 1,
    highlight_models: list = None,
    cmap_name: str = "magma",
    reverse: bool = False,
    use_fixed_range: bool = False,
    fixed_range: tuple = (0.0, 1.0),
    xlim: tuple = (40, 70),
    ylim: tuple = (0.3, 1.0),
    figsize: tuple = (3.5 * 0.66, 2),
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path, usecols=['model_name', f'top{topk}', 'shape_bias'])
    df['shape_bias'] = df['shape_bias'].astype(float)

    x_col = f"top{topk}"
    x_all = df[x_col].values
    y_all = df['shape_bias'].values
    names = df['model_name'].values

    # Sort so high shape bias models draw on top
    order = np.argsort(-y_all)
    x = x_all[order]
    y = y_all[order]
    names = names[order]

    if highlight_models is None:
        highlight_models = []

    is_baseline = np.array([('adult' in name or 'baseline' in name) for name in names])
    is_highlight = np.array([name in highlight_models for name in names])
    is_other = ~is_baseline & ~is_highlight

    cmap = mpl.cm.get_cmap(cmap_name + ("_r" if reverse else ""))
    norm = mpl.colors.Normalize(vmin=fixed_range[0], vmax=fixed_range[1]) if use_fixed_range else mpl.colors.Normalize(vmin=y.min(), vmax=y.max())

    fig, ax = plt.subplots(figsize=figsize)

    # Plot baseline (light gray circles)
    scatter_base = ax.scatter(
        x[is_baseline], y[is_baseline],
        c='lightgray', s=10, edgecolor='k', lw=0, alpha=0.8,
        marker='o', label='Baseline'
    )

    # Plot highlighted models (stars)
    scatter_highlight = ax.scatter(
        x[is_highlight], y[is_highlight],
        c=cmap(norm(y[is_highlight])),
        s=50, edgecolor='k', lw=0.2, alpha=0.99,
        marker='*', label='Highlighted'
    )

    # Plot other models (circles) without legend
    ax.scatter(
        x[is_other], y[is_other],
        c=cmap(norm(y[is_other])),
        s=10, edgecolor='k', lw=0, alpha=0.8,
        marker='o'
    )

    # Colorbar
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, pad=0.03, fraction=0.08)
    cbar.set_label("Shape Bias", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # Axes setup
    ax.set_xlabel(f"Accuracy (top{topk})", fontsize=6)
    ax.set_ylabel("Shape Bias", fontsize=6)
    ax.set_title("Accuracy vs. Shape Bias", fontsize=7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.tick_params(labelsize=5)

    # Add horizontal dashed lines for human benchmarks
    ax.axhline(y=0.96, color='green', linestyle='--', linewidth=1, label='Humans (adult)')
    ax.axhline(y=0.88, color='green', linestyle='--', linewidth=1, label='Humans (4–6 y/o)')


    # Only include legend entries for baseline and highlighted
    handles = [scatter_base, scatter_highlight]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, fontsize=5, framealpha=0.5)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{analysis_name}_colormap_top{topk}.pdf")
    plt.savefig(out)
    plt.close(fig)
    print(f"✔ Saved plot to {out}")
    
#* shape bias - training data size trade  off 
def load_csv(text_or_path: str) -> pd.DataFrame:
    """
    Accept either a filesystem path or a raw multiline CSV string and
    return a cleaned DataFrame.
    """
    if os.path.exists(text_or_path):                        # file on disk
        df = pd.read_csv(text_or_path)
    else:                                                   # raw string
        df = pd.read_csv(StringIO(text_or_path))

    # Convert the 'plotting_color' string "(r,g,b)" → tuple
    # import pdb;pdb.set_trace()
    df["plotting_color"] = df["plotting_color"].apply(
        lambda s:  ast.literal_eval(s)                       # safe tuple parse  [oai_citation:6‡Stack Overflow](https://stackoverflow.com/questions/39150590/literal-eval-of-tuple-with-a-single-element?utm_source=chatgpt.com)
    )

    # Ensure numeric types
    for col in ["avg_shape_bias", "training_data_size", "model_parameter_size_M"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # coerce → NaN → fill later

    # Replace missing parameter sizes with the median
    df["model_parameter_size_M"].fillna(
        df["model_parameter_size_M"].median(), inplace=True
    )
    return df


def plot_shape_bias_data_size_tradeoff(
    csv_path: str,
    highlight=("DVD-P", "DVD-B", "DVD-S"),
    figsize=(3.54*0.65, 2),
    save_path = None,
    annotate: bool = True,              # ← NEW SWITCH
    use_adjusttext: bool = True,          # ← switch off for pure-Matplotlib
    # Define the models to annotate
    annotate_models = [
                'vgg16_bn', 'alexnet', 'resnet50', 'SimCLR: ResNet-50', 
                'ViT-S', 'ViT-B','ViT-L', 'BiT-M: ResNet-101x1 (14M)', 
                'LLaVA-NeXT 7B', 'Qwen-VL Plus', 'Qwen-VL Max', 'InstructBLIP Vicuna-7B', 'InstructBLIP Flan-T5-xl',
                'Emu2-Chat', 
                'DVD-P', 'DVD-B', 'DVD-S', 
                'humans(4-6 y/o)', 'humans(adult)'
            ],
):
    """
    Draw the bubble-scatter.

    Parameters
    ----------
    df         : cleaned DataFrame
    highlight  : iterable of model names to draw as stars
    figsize    : tuple passed to plt.subplots
    save_path    : optional filepath → PDF/PNG/etc.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = load_csv(csv_path)

    # Normalise bubble areas (max → 300 pt²)
    df["bubble_area"] = (
        df["model_parameter_size_M"] / df["model_parameter_size_M"].max() * 300
    )

    is_star = df["model_name"].isin(highlight)

    fig, ax = plt.subplots(figsize=figsize)

    # --- Plain circles ---
    scatter_circles = ax.scatter(
        df.loc[~is_star, "training_data_size"],
        df.loc[~is_star, "avg_shape_bias"],
        s= 10, # df.loc[~is_star, "bubble_area"],
        c=list(df.loc[~is_star, "plotting_color"]),
        alpha=0.8,
        marker="o",
        edgecolor="k",
        linewidth=0.3,
        label="Other models",
    )

    # --- Highlighted stars ---
    scatter_stars = ax.scatter(
        df.loc[is_star, "training_data_size"],
        df.loc[is_star, "avg_shape_bias"],
        s= 50, #df.loc[is_star, "bubble_area"] * 2.5,  # emphasise
        c=list(df.loc[is_star, "plotting_color"]),
        marker="*",
        edgecolor="k",
        linewidth=0.6,
        label="DVD-P/B/S",
    )

    # Axes, grid, legend
    ax.set_xscale("log")                           # log-axis for wide data span  [oai_citation:7‡Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html?utm_source=chatgpt.com)
    ax.set_xlabel("Training-data size") #  (images)/tokens
    ax.set_ylabel("Average shape-bias")
    # ax.set_title("Shape-bias vs training-data size\n(bubble ∝ parameter count)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

     # ── NEW – annotate each point ────────────────────────────
    

    # Iterate over the DataFrame rows
    for _, row in df.iterrows():
        if row['model_name'] in annotate_models:
            # Add annotation with a slight offset
            ax.annotate(
                row['model_name'],
                xy=(row['training_data_size'], row['avg_shape_bias']),
                # xytext=(1, 1),  # Offset position
                # textcoords='offset points',
                fontsize=3,
                ha='left',
                va='bottom',
                # arrowprops=dict(arrowstyle='-', lw=0.3, color='gray')  # Optional: add a line connecting text to point
            )


    plt.tight_layout()                             # fit labels before save  [oai_citation:8‡Stack Overflow](https://stackoverflow.com/questions/16118291/matplotlib-make-final-figure-dimensions-match-figsize-with-savefig-and-bbox-e?utm_source=chatgpt.com)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ saved: {save_path}")
    # plt.show()




class ModelAccuracyPlotter:
    def __init__(self, save_dir='./results/sigma_blur_attack',
                 human_data_path='/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/data/blur_degradation_human_experiment/human_accuracy.mat'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.human_data_path = human_data_path
        self.face_df, self.object_df = self._load_human_data()

    def _load_human_data(self):
        data = loadmat(self.human_data_path)
        # import pdb;pdb.set_trace()
        face_data = self._extract_category_data(data, 'face')
        object_data = self._extract_category_data(data, 'object')
        return face_data, object_data

    def _extract_category_data(self, data, category):
        cat_data = data[category]
        sigmas = np.array(cat_data['sigmas']).flatten()[0].squeeze()
        human_acc = np.array(cat_data['human_accuracy']).T * 100  # Convert to percentage
        human_acc = np.array(human_acc.tolist(), dtype=np.float64).squeeze()
        cnn_acc = np.array(cat_data['cnn_accuracy']).flatten()[0].squeeze() * 100
        blur_cnn_acc = np.array(cat_data['blur_trained_cnn_accuracy']).flatten()[0].squeeze() * 100

        # import pdb;pdb.set_trace()
        df = pd.DataFrame({
            'sigma': sigmas,
            'human_mean': human_acc.mean(axis=0).squeeze(),
            'human_std': human_acc.std(axis=0),
            'cnn': cnn_acc,
            'blur_cnn': blur_cnn_acc
        })
        return df

    @staticmethod
    def logistic_func(x, a, b, c, d):
        return a + (1 - a - b) / (1 + np.exp(-d * (x - c)))

    def fit_logistic(self, x, y):
        popt, _ = curve_fit(self.logistic_func, x, y, p0=[0.5, 0.5, np.median(x), 1], maxfev=5000)
        return popt

    def normalize_accuracy(self, acc_series, absolute=False):
        if absolute:
            return acc_series
        max_val = acc_series.iloc[0]
        return (acc_series / max_val) * 100

    def plot_data_and_fit(self, df, x_fit, y_fit, color, label, marker):
        plt.plot(x_fit, y_fit, '-', linewidth=1, color=color)
        plt.errorbar(df['sigma'], df['human_mean'], yerr=df['human_std'],
                     fmt=marker, linewidth=1, color=color, capsize=0, label=label)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    def plot_model_accuracy(self, new_models, dataset_name, absolute_acc=False, save_name = 'human_and_models_objects_vs_sigma_blur_attack'):
        # Normalize accuracies
        for df in [self.face_df, self.object_df]:
            df['human_mean'] = self.normalize_accuracy(df['human_mean'], absolute_acc)
            df['cnn'] = self.normalize_accuracy(df['cnn'], absolute_acc)
            df['blur_cnn'] = self.normalize_accuracy(df['blur_cnn'], absolute_acc)

        for model_name, acc_list in new_models.items():
            new_models[model_name] = self.normalize_accuracy(pd.Series(acc_list), absolute_acc)

        # cmap = cm.get_cmap('Set1', 9)
        colors =  [black, teal_green,  bright_green, deep_blue, 'red', 'purple'] #['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'yellow']
        x_fit = np.linspace(0, 32, 320)

        if 'face' in dataset_name:
            # Plot for Faces
            plt.figure(figsize=(3.54*0.8, 2*1.2 )) # * 1.2
            popt = self.fit_logistic(self.face_df['sigma'], self.face_df['human_mean'])
            y_fit = self.logistic_func(x_fit, *popt)
            self.plot_data_and_fit(self.face_df, x_fit, y_fit, 'red', 'Human', '^')

            for i, (model_name, model_data) in enumerate(new_models.items()):
                model_acc = model_data[:len(self.face_df)]
                popt = self.fit_logistic(self.face_df['sigma'], model_acc)
                y_fit = self.logistic_func(x_fit, *popt)
                plt.plot(x_fit, y_fit, '-', linewidth=1, color=colors[i % len(colors)])
                plt.plot(self.face_df['sigma'], model_acc, 's', color=colors[i % len(colors)], label=model_name)

            plt.axhline(10, linestyle=':', color='k')  # Chance level
            plt.xlabel('Blur level')
            plt.ylabel('Accuracy (%)')
            plt.xticks(self.face_df['sigma'], ['0', '', '', '', '8', '', '16', '', '24', '32'])
            plt.yticks(np.arange(0, 101, 25))
            plt.xlim(-1, 33)
            plt.ylim(0, 100)
            plt.title('Faces')
            plt.grid(False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{save_name.replace('object', 'face')}.pdf")

        else:
            # Plot for Objects
            plt.figure(figsize=(3.54*0.8, 2*1.2))
            popt = self.fit_logistic(self.object_df['sigma'], self.object_df['human_mean'])
            y_fit = self.logistic_func(x_fit, *popt)
            self.plot_data_and_fit(self.object_df, x_fit, y_fit, 'red', 'Human', '^')

            for i, (model_name, model_data) in enumerate(new_models.items()):
                model_acc = model_data[:len(self.object_df)]
                popt = self.fit_logistic(self.object_df['sigma'], model_acc)
                y_fit = self.logistic_func(x_fit, *popt)
                plt.plot(x_fit, y_fit, '-', linewidth=1, color=colors[i % len(colors)])
                plt.plot(self.object_df['sigma'], model_acc, 'o', color=colors[i % len(colors)], label=model_name)

            plt.axhline(6.25, linestyle=':', color='k')  # Chance level
            plt.xlabel('Blur level')
            plt.ylabel('Accuracy (%)')
            plt.xticks(self.object_df['sigma'], ['0', '', '', '', '8', '', '16', '', '24', '32'])
            plt.yticks(np.arange(0, 101, 25))
            plt.xlim(-1, 33)
            plt.ylim(0, 100)
            plt.title('Objects')
            plt.grid(False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/{save_name}.pdf')
            # plt.show()

if __name__ == "__main__":
    # Example usage
    csv_file_path = '/home/student/l/lzejin/codebase/P001_dvd_gpus/results/shape_bias/19th_Wed_traverse_4_shape_bais_plot_1.csv'
    plot_shape_bias_across_time(csv_file_path, save_path='/home/student/l/lzejin/codebase/P001_dvd_gpus/results/plots/shape_bias/shape_bias_plot_B2.pdf')
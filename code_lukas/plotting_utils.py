import matplotlib.pyplot as plt

def cm_to_inches(cm_tuple):
    return tuple(x / 2.54 for x in cm_tuple)

def setup_plotting_environment():
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 6, 'font.family': 'Helvetica'})

def plot_overview_labels_along_y_axis(fig, ax, labels, vals_per_label, x_offset, no_y_vals,
                                      fontsize=7, fontweight="bold", color="black", linewidth=1, stagger=0,
                                      rotation=90):
    """
    Plot labels along the y-axis with vertical lines to group related items.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object to plot on
    labels : list or array-like
        List of label strings to display along the y-axis
    vals_per_label: dict or pandas.Series
        Dictionary/Series mapping each label to the number of values it represents.
        Keys should match entries in 'labels'
    x_offset : float
        Horizontal position offset for placing the labels (negative values place left of y-axis)
    no_y_vals : int
        Total number of y-axis values/positions to distribute labels across
    fontsize : int, optional (default=7)
        Font size for the label text
    fontweight : str, optional (default="bold")
        Font weight for the label text ("bold", "normal", etc.)
    color : str, optional (default="black")
        Color for both text and lines
    linewidth : float, optional (default=1)
        Width of the vertical grouping lines
    stagger : float, optional (default=0)
        Horizontal staggering offset for alternating labels (to avoid overlap)
    rotation : float, optional (default=90)
        Rotation angle for the label text in degrees
    """
    # Get the y-axis tick positions
    y_ticks = ax.get_yticks()
    
    # Get the first and last tick positions
    y_top = y_ticks[-1]
    y_bottom = y_ticks[0]
    
    total_distance = y_top - y_bottom
    step_per_label = total_distance / (no_y_vals - 1)
    
    # Freeze the current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    cumulative_vals = 0
    for index, label in enumerate(labels):
        y_pos = y_top - step_per_label * (cumulative_vals + vals_per_label[label]/2 - 0.5)
        curr_x_val = x_offset + stagger * (index % 2)
        
        if not "empty" in label and vals_per_label[label] > 0:
            # Add text annotations
            ax.text(
                curr_x_val,
                y_pos,
                label,
                ha='right',
                va='center',
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                rotation=rotation,
                transform=ax.transData
            )
            
            adjustment_for_single_value = 0.4 if vals_per_label[label] == 1 else 0
            
            # Draw vertical line
            ax.plot(
                [curr_x_val + 0.2, curr_x_val + 0.2],
                [y_top - step_per_label * (cumulative_vals) + adjustment_for_single_value, 
                y_top - step_per_label * (cumulative_vals + vals_per_label[label] - 1) - adjustment_for_single_value],
                color=color,
                linewidth=linewidth,
                clip_on=False,
                zorder=0  # Ensures it doesn't interfere with other plot elements
            )
        
        cumulative_vals += vals_per_label[label]
    
    # Restore original plot limits to prevent them from changing
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
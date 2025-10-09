from matplotlib.ticker import FuncFormatter

# Greek letters mapping
_greek_letters = {
    "-α": "-alpha",
    "α-": "alpha-", 
    " α ": " alpha ",
    "-β": "-beta",
    "β-": "beta-",
    " β ": " beta ",
    "-γ": "-gamma",
    "γ-": "gamma-",
    " γ ": " gamma ",
    "-δ": "-delta",
    "δ-": "delta-",
    " δ ": " delta ",
    "-ε": "-epsilon",
    "ε-": "epsilon-",
    " ε ": " epsilon ",
    "-ζ": "-zeta",
    "ζ-": "zeta-",
    " ζ ": " zeta ",
    "-η": "-eta",
    "η-": "eta-",
    " η ": " eta ",
    "-θ": "-theta",
    "θ-": "theta-",
    " θ ": " theta ",
    "-ι": "-iota",
    "ι-": "iota-",
    " ι ": " iota ",
    "-κ": "-kappa",
    "κ-": "kappa-",
    " κ ": " kappa ",
    "-λ": "-lambda",
    "λ-": "lambda-",
    " λ ": " lambda ",
    "-μ": "-mu",
    "μ-": "mu-",
    " μ ": " mu ",
    "-ν": "-nu",
    "ν-": "nu-",
    " ν ": " nu ",
    "-ξ": "-xi",
    "ξ-": "xi-",
    " ξ ": " xi ",
    "-ο": "-omicron",
    "ο-": "omicron-",
    " ο ": " omicron ",
    "-π": "-pi",
    "π-": "pi-",
    " π ": " pi ",
    "-ρ": "-rho",
    "ρ-": "rho-",
    " ρ ": " rho ",
    "-σ": "-sigma",
    "σ-": "sigma-",
    " σ ": " sigma ",
    "-τ": "-tau",
    "τ-": "tau-",
    " τ ": " tau ",
    "-υ": "-upsilon",
    "υ-": "upsilon-",
    " υ ": " upsilon ",
    "-φ": "-phi",
    "φ-": "phi-",
    " φ ": " phi ",
    "-χ": "-chi",
    "χ-": "chi-",
    " χ ": " chi ",
    "-ψ": "-psi",
    "ψ-": "psi-",
    " ψ ": " psi ",
    "-ω": "-omega",
    "ω-": "omega-",
    " ω ": " omega ",
}

# Reverse mapping
_greek_letters_inv = {v: k for k, v in _greek_letters.items()}

def replace_greek_letters(text):
    for key, value in _greek_letters.items():
        text = text.replace(key, value)
    return text

def introduce_greek_letters(text):
    """
    Replace spelled-out Greek letters with their symbol.
    """
    for key, value in _greek_letters_inv.items():
        text = text.replace(key, value)
    return text

def apply_greek_labels(ax):
    """
    Apply Greek letter formatting to axis tick labels.
    Numeric labels are preserved or formatted; text labels are processed.
    """
    original_x_formatter = ax.xaxis.get_major_formatter()
    original_y_formatter = ax.yaxis.get_major_formatter()

    def custom_tick_formatter(original_formatter, value, tick_position, is_numeric_axis=False):
        """
        Custom formatter that applies Greek letters to text and preserves/formats numbers.
        """
        label_string = ""
        # Ensure the original_formatter is callable before calling it
        if callable(original_formatter):
            label_string = original_formatter(value, tick_position)
        
        # If the original formatter gave a non-empty string...
        if label_string:
            try:
                # Test if it's a number. If so, return it directly.
                float(label_string.strip())
                return label_string
            except ValueError:
                # Not a number, apply Greek letter substitution.
                return introduce_greek_letters(label_string)
        # If original returned empty, AND it's a numeric axis AND value is a number...
        elif is_numeric_axis and isinstance(value, (int, float)):
            # Fallback: Format the number yourself.
            # Use 'g' for a general format or specify (e.g., '.1f').
            return f"{value:g}"
        # Otherwise (text axis with empty original, or non-number), return empty.
        else:
            return ""

    # Apply the custom formatter, telling the Y-axis it's likely numeric.
    # Note: We assume Y is numeric and X might be text/category.
    # If X can also be purely numeric and disappear, set is_numeric_axis=True for it too.
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: custom_tick_formatter(original_x_formatter, val, pos, is_numeric_axis=False)
    ))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: custom_tick_formatter(original_y_formatter, val, pos, is_numeric_axis=True)
    ))

    # Redraw the figure to apply formatter changes.
    if ax.figure and ax.figure.canvas:
        ax.figure.canvas.draw_idle()
    

def apply_greek_labels_heatmap(ax):
    """
    Replace Greek letter names with symbols in x and y tick labels for heatmaps.
    """
    # Apply to x-axis labels
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticklabels([introduce_greek_letters(lbl) for lbl in x_labels], rotation=90)
    
    # Apply to y-axis labels
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticklabels([introduce_greek_letters(lbl) for lbl in y_labels])
    
    # Ensure the figure updates
    ax.figure.canvas.draw_idle()
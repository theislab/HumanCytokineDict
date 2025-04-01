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
    Apply Greek letter formatting to both x and y axis tick labels on a given axes.
    """
    def greek_label_formatter(val, pos):
        label = f"{val:.0f}"  # Format as string
        return introduce_greek_letters(label)
    
    # Apply formatter to x and y axes
    ax.xaxis.set_major_formatter(FuncFormatter(greek_label_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(greek_label_formatter))
    
    # Redraw the figure to ensure updates are visible
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
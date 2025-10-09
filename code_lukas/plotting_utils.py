import matplotlib.pyplot as plt

def cm_to_inches(cm_tuple):
    return tuple(x / 2.54 for x in cm_tuple)

def setup_plotting_environment():
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 6, 'font.family': 'Helvetica'})

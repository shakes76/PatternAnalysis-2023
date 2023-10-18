'''Shows example usage of trained model. Print out any results and/ or provide visualisations where applicable'''
import matplotlib.pyplot as plt

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
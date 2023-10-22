# Shows example usage of my trained model. Print out any results and / or provide visualisations where applicable
from modules import *
from dataset import *

def save_plot(iteration, loss, epoch, title):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('iteration_loss_' + str(epoch) + '.png')
    
def save_plot_acc(iteration, loss, epoch, title, xtitle, ytitle):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(str(epoch) + '.png')

def show_plot(iteration, loss, title):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
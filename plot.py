import matplotlib.pyplot as plt
import os
from collections import namedtuple

Losses = namedtuple('losss', ('D_loss', 'G_loss'))

class Plots(object):
    def __init__(self, name):
        self.counter = 0
        self.name = name

    def plot_rewards(self, losses, save=False):
        x = [i for i in range(len(losses))]

        if self.counter == 0:
            # plt.ion()
            # plt.show()
            pass

        losses = Losses(*zip(*losses))

        d_losses = losses.D_loss
        g_losses = losses.G_loss
        
        plt.plot(x, d_losses, label='Descriminator')
        plt.plot(x, g_losses, label='Generator')
        plt.xlabel('Training iteration')
        plt.ylabel('Losses')
        # plt.draw()
        # plt.pause(0.001)

        if save:
            if not os.path.isdir('./figures/'):
                os.mkdir('./figures/')

            plt.savefig('./figures/' + self.name + '.png')
        plt.close()

def plot_gan_loss(losses, directory, save=True):
    x = [i for i in range(len(losses))]
    losses = Losses(*zip(*losses))

    d_losses = losses.D_loss
    g_losses = losses.G_loss
    
    plt.plot(x, d_losses, label='Descriminator')
    plt.plot(x, g_losses, label='Generator')
    plt.xlabel('Training iteration')
    plt.ylabel('Losses')
    plt.savefig(directory + 'GAN_losses.png')
    plt.close()
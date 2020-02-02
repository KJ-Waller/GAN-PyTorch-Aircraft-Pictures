import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from plot import Plots
from plot import plot_gan_loss
# import multiprocessing
from gan_models import Descriminator, Generator
from dataset import get_dataloader
from PIL import Image
# multiprocessing.set_start_method('spawn', True)


class GAN(object):
    def __init__(self, version):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.lr = 5e-5
        
        self.version = version

        self.beta = .5
        
        self.D_net = Descriminator(self.lr, self.beta).to(self.device)
        self.G_net = Generator(self.lr, self.beta).to(self.device)
        
        self.batch_size = 64
        self.dataloader = get_dataloader(batch_size=self.batch_size)

        self.z_constant = torch.randn(64, 100, 1, 1)
        self.gen_image_counter = 0
        
        self.loss = nn.BCELoss()
        self.losses = []
        self.fixed_gen_images = []

        self.fixed_z = torch.randn(self.batch_size, 100, 1, 1).to(self.device)

        self.create_directories()

        self.load_models()

    def create_directories(self):
        self.root_dir = './model_' + self.version + '/'
        self.model_dir = self.root_dir + 'models/'
        self.figure_dir = self.root_dir + 'figures/'
        self.const_image_dir = self.root_dir + 'gen_constant_images/'
        
        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.isdir(self.figure_dir):
            os.mkdir(self.figure_dir)
        if not os.path.isdir(self.const_image_dir):
            os.mkdir(self.const_image_dir)

    def save_models(self):
        torch.save(self.G_net.state_dict(), self.model_dir + 'generator_model_' + self.version + '.pth')
        torch.save(self.D_net.state_dict(), self.model_dir + 'descriminator_model_' + self.version + '.pth')
        print(f'Models saved to {self.model_dir} folder')

    def load_models(self):
        try:
            self.G_net.load_state_dict(torch.load(self.model_dir + 'generator_model_' + self.version + '.pth'))
            self.D_net.load_state_dict(torch.load(self.model_dir + 'descriminator_model_' + self.version + '.pth'))
            print(f'Pretrained models found and loaded')
        except:
            print('No pretrained models found, creating new models')

    def gen_const_images(self):
        with torch.no_grad():
            gen_images = self.G_net(self.z_constant.to(self.device)).to(self.device).detach().cpu()
        gen_images_grid = utils.make_grid(gen_images, padding=2, normalize=True)
        plt.axis("off")
        plt.title("Fake Images")
        plt.figure(figsize=(15,15))

        image = np.transpose(gen_images_grid,(1,2,0)).numpy()
        plt.imsave(self.const_image_dir + 'gen_constant_' + str(self.gen_image_counter) + '.png', image)
        self.gen_image_counter += 1

        # TODO: Generate a gif
        self.fixed_gen_images.append(image)

        frames = [Image.fromarray(frame.astype(dtype=np.uint8)) for frame in self.fixed_gen_images]
        frames[0].save(self.const_image_dir + 'gen_images_progress.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)



    def plot_loss(self):
        plot_gan_loss(self.losses, self.figure_dir)
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for idx, batch in enumerate(self.dataloader):

                # 1) Train the descriminator with real and fake/generated images
                # Set Generator to eval, and Descriminator to train
                # self.G_net.eval()
                # self.D_net.train()

                # Zero grad descriminator, otherwise we accumulate gradients
                self.D_net.zero_grad()

                # First pass real images through descriminator
                batch = batch[0].to(self.device)
                real_samples_values = self.D_net(batch).view(-1)

                # Calculate loss, backward propagate and do one optimizer step on the Descriminator
                # real_labels = torch.ones(batch.size(0)).to(self.device)
                real_labels_noise = torch.FloatTensor(batch.size(0)).uniform_(0.9, 1).to(self.device)
                D_real_loss = self.loss(real_samples_values, real_labels_noise)
                D_real_loss.backward()
                
                # Now generate fake images by passing a randomally generated
                # latent variable z through the generator, and pass fake images through the descriminator
                z = torch.randn(self.batch_size, 100, 1, 1).to(self.device)
                fake_samples = self.G_net(z)
                fake_samples_values = self.D_net(fake_samples.detach()).view(-1)
                # fake_labels = torch.zeros(self.batch_size).to(self.device)
                fake_labels_noise = torch.FloatTensor(self.batch_size).uniform_(0, 0.1).to(self.device)
                D_fake_loss = self.loss(fake_samples_values, fake_labels_noise)
                D_fake_loss.backward()
                D_loss = D_real_loss + D_fake_loss

                # Now that the loss for both real and fake images has been calculated,
                # we perform one optimizer step on the descriminator
                self.D_net.optim.step()

                # 2) Train the generator
                # Set Generator to train, and Descriminator to eval
                # self.G_net.train()
                # self.D_net.eval()

                # Zero grad generator
                self.G_net.zero_grad()

                # Generate fake images, and pass them through descriminator network
                fake_samples_values = self.D_net(fake_samples).view(-1)

                # The goal of the generator is to fool the descriminator, so the target labels are 1 (the real image labels)
                target_labels = torch.ones(self.batch_size).to(self.device)

                # Calculate loss, backward propagate, and do one optimizer step
                G_loss = self.loss(fake_samples_values, target_labels)
                G_loss.backward()

                self.G_net.optim.step()

                # Log and print the losses
                self.losses.append((D_loss.item(), G_loss.item()))

                # Every 10 iterations, generate fake images using fixed latent z
                if (epoch+idx) % 100 == 0:
                    print(f'Epoch {epoch}/{num_epochs}, iter {idx}\n\tD_loss: {D_loss}\n\tG_loss: {G_loss.item()}')
                    self.save_models()
                    # self.plotter.plot_rewards(self.losses, True)
                    self.plot_loss()
                    self.gen_const_images()

gan = GAN('v7_planes_256')
gan.train(50)
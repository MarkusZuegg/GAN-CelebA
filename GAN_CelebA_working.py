import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import MNIST, CelebA
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#Initiate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Batch size during training
BATCH_SIZE = 128

def show_tensor_images(image_tensor, num_images=25, size=(3, 64, 64), ret=False):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    if ret:
        return image_grid.permute(1, 2, 0).squeeze()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig('./lightning_logs/faces.png')

def weights_init(m):
    """Initilises weights for networks
        mean: 0 or 1
        STD: 0.02"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_noise(cur_batch_size, z_dim):
    """Creates random noise tensor
        Shape = (batch_size, z_dim, 1, 1)"""
    noise = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
    return noise

class Generator(nn.Module):
    """Generator Network for GAN
        Creates 5 blocks of ConvTranspose2d layers
        Given a vector of z_dim generate an image"""
    def __init__(self, in_channels=3, z_dim=100):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.create_block(z_dim, 1024, kernel_size=4, stride=1, padding=0),
            self.create_block(1024, 512, kernel_size=4, stride=2, padding=1),
            self.create_block(512, 256, kernel_size=4, stride=2, padding=1),
            self.create_block(256, 128, kernel_size=4, stride=2, padding=1),
            self.create_block(128, 3, kernel_size=4, stride=2, padding=1, final_layer=True),)

    def create_block(self, in_channels, out_channels, kernel_size=5, 
                       stride=2, padding=1, final_layer=False):
        """Creates a layer of Convtransposed2d neural nets"""
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                   stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True))

    def forward(self, noise):
        return self.gen(noise)
     
class Discriminator(nn.Module):
    """Discriminator Network to train GAN
        Creates a neural net of 5 conv2d blocks
        Use is to decide if given image is fake or real"""
    def __init__(self, in_channels=3, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(in_channels, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, stride=1),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4, stride=2),
            self.make_disc_block(hidden_dim * 4, 1, final_layer=True),)

    def make_disc_block(self, input_channels, output_channels,
                        kernel_size=4, stride=2, final_layer=False):
        """Creates a layer of convolution sequence"""
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride))
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size,stride), 
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2))

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
    
class GAN(LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 100,
        hidden_dim: int = 32,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.latent_dim = latent_dim

        #networks
        self.generator = Generator(in_channels, z_dim=latent_dim)
        self.discriminator = Discriminator(in_channels=in_channels, hidden_dim=hidden_dim)

        #apply weights to networks
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    
    def generator_step(self, x, noise):
        """Logs and returns the loss for generator
            x: real image"""
        #generate fake images
        fake_images = self.generator(noise)

        fake_logits = self.discriminator(fake_images)
        fake_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))

        gen_loss = fake_loss

        self.log('gen_loss', gen_loss, on_epoch=True, prog_bar=True)
        return gen_loss
    
    def discriminator_step(self, x, noise):
        """Logs and returns the loss for generator
            x: real image"""

        fake_images = self.generator(noise)

        #get discriminator outputs
        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(fake_images.detach())

        #real loss
        real_loss = self.adversarial_loss(real_logits, torch.ones_like(real_logits))
        #fake loss
        fake_loss = self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        disc_loss = (fake_loss + real_loss) / 2

        self.log('disc_loss', disc_loss, on_epoch=True, prog_bar=True)
        return disc_loss

    def training_step(self, batch, batch_idx):
        #get optimizers
        g_opt, d_opt = self.optimizers()

        #read in batch
        imgs, _ = batch
        real = imgs

        #sample noise
        noise = get_noise(real.shape[0], self.latent_dim)

        # generate images for logging
        self.generated_imgs = self.generator(noise)

        # log sampled images
        sample_imgs = self.generated_imgs[:18]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        if batch_idx % 500 == 0:
            show_tensor_images(image_tensor=self.generated_imgs,)

        #get generator loss
        g_loss = self.generator_step(real, noise)

        #Manually step the generator optimizer
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        #get discriminator loss
        d_loss = self.discriminator_step(real, noise)

        #Manually step discriminator optimizer
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        #return generator and discriminator loss into dict
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        """Configures optimizers for each step"""
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return g_opt, d_opt

    # def validation_step(self,batch,batch_idx):
    #     imgs, _ = batch
    #     noise = get_noise(imgs.shape[0], self.latent_dim)

    #     # log sampled images
    #     sample_imgs = self.generator(noise)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    
class CelebADataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size = BATCH_SIZE,
        image_size = 64,
        in_channels = 3):
        super().__init__()
        self.batch_size = batch_size
        self.num_works = 4

        #transform for CelebA dataset
        self.transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        self.dims = (in_channels, image_size, image_size)

        #load data
        self.dataset = torchvision.datasets.ImageFolder(root='./CelebA_data/celeba/', transform=self.transform)

        #split data
        lengths = [170000, 30000, 2599]
        self.train, self.val, self.test = random_split(dataset=self.dataset, lengths=lengths)

        #Confirm dataloading correctly
        print('Data loaded')


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_works)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_works)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_works)
    
def main():
    max_epochs = 50

    #load pylightning datamodule
    data = CelebADataModule()
    #load pylightning model
    model = GAN()
    #set up trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],)
    #run model agaisnt datamodule
    trainer.fit(model, data)

if __name__ == '__main__': main()
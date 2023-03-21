import os
import math
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_tensor
import torchvision.datasets
import typer
from sliced_wasserstein import LinearProjector, NNProjector, GSWD, MGSWD
from fid_score import evaluate_fid_score
from torch.optim.lr_scheduler import _LRScheduler


def get_projector(proj_type, input_features, final_dim, hidden_dim=None):
    if proj_type == 'LinearProjector':
        projector = LinearProjector(input_features, final_dim)
    else:
        projector = NNProjector(
            num_layer=2, input_features=input_features,
            hidden_dim=hidden_dim if hidden_dim else final_dim, final_dim=final_dim
        )
    return projector

class CosineAnnealingWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(CosineAnnealingWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            cos_out = (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) / 2
            return [self.eta_min + (base_lr - self.eta_min) * cos_out for base_lr in self.base_lrs]


class Logger:
    def __init__(self, experiment_name):
        base_folder_name = 'results'
        self.experiment_folder = f'{base_folder_name}/{experiment_name}/{datetime.now().strftime("%b%d_%H-%M-%S")}'
        self.ae_samples_folder = f'{self.experiment_folder}/ae_samples'
        self.sw_samples_folder = f'{self.experiment_folder}/sw_samples'
        os.makedirs(self.ae_samples_folder)
        os.makedirs(self.sw_samples_folder)
        self.ae_fid = open(f'{self.experiment_folder}/ae_fid', 'a')
        self.sw_fid = open(f'{self.experiment_folder}/sw_fid', 'a')

    def write_results(
        self, gen_images, target_images, loss, epoch, ae_or_sw
    ):
        folder = getattr(self, f'{ae_or_sw}_samples_folder')
        file_writer = getattr(self, f'{ae_or_sw}_fid')
        with torch.no_grad():
            comparison = torch.empty(100, 1, *gen_images.shape[2:], dtype=gen_images.dtype)
            comparison[::2] = gen_images[:50].cpu()
            comparison[1::2] = target_images[:50].cpu()
            samples = make_grid(comparison.cpu(), nrow=10, normalize=False)

            save_image(samples,
                       f'{folder}/samples_{epoch}.png')
            fid = evaluate_fid_score(
                gen_images.cpu(), target_images.cpu(), batch_size=128
            )
            file_writer.write(f'Epoch {epoch}: {fid}\n')
        typer.echo(f'{ae_or_sw.upper()} Loss Epoch {epoch}: {loss}')
        file_writer.flush()

    def write_ae_results(self, gen_images, target_images, loss, epoch):
        self.write_results(gen_images, target_images, loss, epoch, 'ae')

    def write_sw_results(self, ae, z, target_images, loss, epoch):
        with torch.no_grad():
            gen_images = ae.decode(z)
        self.write_results(gen_images, target_images, loss, epoch, 'sw')

    def close(self):
        self.ae_fid.close()
        self.sw_fid.close()


class MnistEncoder(nn.Module):
    def __init__(self, final_dim):
        super(MnistEncoder, self).__init__()
        self.final_dim = final_dim

        self.convs = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(24, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=4, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
        )

        self.linear = nn.Sequential(
            nn.Linear(49*16, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, self.final_dim)
        )

    def forward(self, x):
        return self.linear(self.convs(x).flatten(1))


class MnistDecoder(nn.Module):
    def __init__(self, initial_dim):
        super(MnistDecoder, self).__init__()
        self.initial_dim = initial_dim
        self.linear = nn.Sequential(
            nn.Linear(self.initial_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 49*16)
        )

        upsample = nn.UpsamplingNearest2d
        self.convs = nn.Sequential(
            nn.Conv2d(49, 49, kernel_size=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(49, 25, kernel_size=3, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(25, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convs(self.linear(x).view(x.shape[0], -1, 4, 4))


class MnistClassifier(nn.Module):
    def __init__(self, initial_dim):
        super(MnistClassifier, self).__init__()
        self.inital_dim = initial_dim

        self.linear = nn.Sequential(
            nn.Linear(self.inital_dim, 2*self.inital_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2*self.inital_dim, 4*self.inital_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4*self.inital_dim, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.linear(x.flatten(1))


class AE(nn.Module):
    def __init__(self, dim=1):
        super(AE, self).__init__()
        self.dim = dim
        self.encoder = MnistEncoder(final_dim=dim)
        self.decoder = MnistDecoder(initial_dim=dim)

    @classmethod
    def load(cls, path):
        d = torch.load(path)
        instance = cls.__new__(cls)
        instance.__init__(*d['args'], **d['kwargs'])
        instance.load_state_dict(d['state_dict'])
        return instance

    @property
    def device(self):
        return self.encoder.convs[0].weight.device

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, x):
        with torch.no_grad():
            return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def save(self, folder):
        self.cpu()
        name = 'autoencoder.pt'
        torch.save({'state_dict': self.state_dict(),
                    'type': 'AE',
                    'args': [],
                    'kwargs': {'dim': self.dim}
                    }, f'{folder}/{name}')

    def train_ae(self, epochs, dataloader, logger):
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=1e-5
        )
        scheduler = CosineAnnealingWarmup(
            optimizer=optimizer, total_epochs=epochs*len(dataloader),
            eta_min=0.0003, warmup_epochs=1000
        )
        for epoch in range(epochs):
            for images, label in dataloader:
                images = images.to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                z = self.encoder(images)
                gen_images = self.decoder(z)
                loss = nn.functional.binary_cross_entropy(gen_images, images)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if (epoch + 1) % 10 == 0:
                if (epoch + 1) % 100 == 0:
                    logger.write_ae_results(
                        gen_images, images, loss.item(), epoch
                    )

        self.eval()

    def test_ae(self, dataloader, logger):
        self.eval()
        with torch.no_grad():
            images, label = next(iter(dataloader))
            images = images.to(self.device)
            gen_images = self.decode(self.encode(images))
            loss = nn.functional.binary_cross_entropy(gen_images, images)
        logger.write_ae_results(gen_images, images, loss.item(), 'test')


class LatentGenerator(nn.Module):
    def __init__(self, dim):
        super(LatentGenerator, self).__init__()
        self.dim = dim
        self.linear_layers = nn.Sequential(
            nn.Linear(self.dim//4, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, self.dim),
        )

    def forward(self, batch_size=1):
        device = self.linear_layers[0].weight.data.device
        z = torch.randn(batch_size, self.dim//4, device=device)
        return self.linear_layers(z)


def main(
        gsw_type: str,
        proj_type: str,
        batch_size: int = 2048,
        lr: float = 5e-3,
        experiment_name: str = 'mnist_test',
        epochs: int = 200,
        num_projections: int = 1000,
        device: str = 'cuda:0',
        autoencoder_path: str = '',
        loss_type: str = 'mse',
        autoencoder_dim: int = 4,
):
    # Setup Log Directory
    logger = Logger(experiment_name=experiment_name)

    # Setup Loss Objective
    assert proj_type in ('LinearProjector', 'NNProjector')
    input_features = autoencoder_dim  # Features used for projector

    assert gsw_type in ('GSWD', 'MGSWD')
    if gsw_type == 'GSWD':
        typer.echo('Running generalized sliced wasserstein')
        projector = get_projector(proj_type, input_features, num_projections)
        sw_loss = GSWD(projector, loss_type)
    else:
        # override num_projections. We only need 1 direction for max
        typer.echo('Running max generalized sliced wasserstein')
        projector = get_projector(proj_type, input_features, final_dim=1, hidden_dim=num_projections)
        sw_loss = MGSWD(
            projector=projector, loss_type=loss_type, lr=1, iterations=10
        )

    # Setup Data
    train_dataset = torchvision.datasets.MNIST(
        '~/.torch/data', train=True, transform=to_tensor
    )
    test_dataset = torchvision.datasets.MNIST(
        '~/.torch/data', train=False, transform=to_tensor
    )
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=12,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=12,
    )

    # Setup Model
    if autoencoder_path == "":
        ae = AE(dim=autoencoder_dim).cuda(device)
        ae.train_ae(700, dataloader, logger)
        ae.test_ae(test_dataloader, logger)
        ae.save(logger.experiment_folder)
    else:
        ae = AE.load(autoencoder_path)
    ae.requires_grad_(False)
    ae.eval()

    generator = LatentGenerator(dim=autoencoder_dim)

    # Setup Training
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    total_epochs = epochs*len(dataloader)
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=total_epochs/30,
                                      total_epochs=total_epochs)
    assert len(dataloader) == len(dataloader.dataset) // batch_size + (1 if len(dataloader.dataset)%batch_size else 1)

    ae.cuda(device)
    projector.cuda(device)
    generator.cuda(device)
    generator.train()

    for epoch in range(epochs):
        for iteration, (image, _) in enumerate(dataloader):
            image = image.cuda(device)
            optimizer.zero_grad()
            z = generator(image.shape[0])
            with torch.no_grad():
                target_z = ae.encoder(image)
            loss = sw_loss(z.flatten(1), target_z.flatten(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
        if (epoch+1) % 100 == 0:
            with torch.no_grad():
                z = generator(image.shape[0])
            logger.write_sw_results(ae, z, image, loss.item(), epoch)

    generator.eval()

    # Quick Test
    with torch.no_grad():
        images, label = next(iter(test_dataloader))
        images = images.to(ae.device)
        target_z = ae.encoder(images)
        z = generator(images.shape[0])
        loss = sw_loss(z.flatten(1), target_z.flatten(1))
        logger.write_sw_results(ae, z, images, loss.item(), 'test')

    # Save Generator
    torch.save(generator.cpu().state_dict(),
               f'{logger.experiment_folder}/generator.pt')

    logger.close()


if __name__ == '__main__':
    typer.run(main)

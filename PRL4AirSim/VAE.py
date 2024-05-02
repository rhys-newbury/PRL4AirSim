import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import transforms


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(
            num_output_channels=1
        ),  # Convert to grayscale with one channel
        # Add more transformations if required
    ]
)



class Encoder(nn.Module):

    def __init__(self, input_dim=4096, hidden_dim=512, latent_dim=256):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, output_dim=4096, hidden_dim=512, latent_dim=256):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))
        return x_hat


class VAE(nn.Module):

    def __init__(self, input_dim=4096, hidden_dim=400, latent_dim=32, device="cpu"):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.conv_output_dim = latent_dim

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def get_latent(self, x):
        mean, logvar = self.encode(x)
        return self.reparameterization(mean, logvar)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train(train_loader, model, optimizer, epochs, device, x_dim=784):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            # print(x.shape)
            # import pdb

            # pdb.set_trace()
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch",
            epoch + 1,
            "\tAverage Loss: ",
            overall_loss / (batch_idx * batch_size),
        )
    return overall_loss


def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(64, 64)  # reshape vector to 2d array
    plt.title(f"[{mean},{var}]")
    plt.imshow(digit, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_latent_space(model, scale=5.0, n=25, digit_size=64, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":

    # download the MNIST datasets
    # path = "~/datasets"
    # train_dataset = MNIST(path, transform=transform, download=True)
    # test_dataset = MNIST(path, transform=transform, download=True)
    # print(len(train_dataset))
    path = "/home/taco/repos/PRL4AirSim/PRL4AirSim/pend_images"

    # Define transformations if needed
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(
                num_output_channels=1
            ),  # Convert to grayscale with one channel
            # Add more transformations if required
        ]
    )

    # Load the dataset
    pend_dataset = ImageFolder(path, transform=transform)

    # Define the sizes of train and test sets
    train_size = int(0.8 * len(pend_dataset))  # 80% for training
    test_size = len(pend_dataset) - train_size  # Remaining for testing

    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(pend_dataset, [train_size, test_size])

    # create train and test dataloaders
    batch_size = 100
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get 25 sample training images for visualization
    dataiter = iter(train_loader)
    image = next(dataiter)

    num_samples = 25
    sample_images = [image[0][i, 0] for i in range(num_samples)]

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap="gray")
        ax.axis("off")

    plt.show()

    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(train_loader, model, optimizer, epochs=50, device=device, x_dim=4096)
    torch.save(model.state_dict(), "./VAE_32.pt")
    # model.load_state_dict(torch.load("./VAE.pt"))
    generate_digit(0.0, 1.0)
    generate_digit(1.0, 0.0)
    plot_latent_space(model, scale=1.0)
    plot_latent_space(model, scale=5.0)

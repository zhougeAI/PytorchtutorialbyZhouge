import torch
import torch.nn as nn
from networkutils import Classifier,ConvTranspose_layer
from torchsummary import summary
from tensorboardX import SummaryWriter

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Classifier()

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.code_linear2 = nn.Linear(in_features=100, out_features=25 * 25 * 64)
        self.code_activation2 = nn.ELU()
        # decoder layer 1
        self.decoder1 = ConvTranspose_layer(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        # decoder layer 2
        self.decoder2 = ConvTranspose_layer(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        # decoder layer 3
        self.decoder3 = ConvTranspose_layer(in_channels=32, out_channels=3, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.code_linear2(x)
        x = self.code_activation2(x)
        x = x.view(-1, 64, 25, 25)
        x = self.decoder1(x)
        x = self.decoder2(x)
        output = self.decoder3(x)
        return output


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)

        return output, embedding

def visualize_classifier():
    classifier1 = Classifier()
    summary(model=classifier1, input_size=[(3, 100, 100)], batch_size=1, device='cpu')

def visualize_decoder():
    decoder = Decoder()
    summary(model=decoder, input_size=[(1, 100)], batch_size=1, device='cpu')

def visualize_autoencoder():
    autoencoder = Autoencoder()
    summary(model= autoencoder, input_size=[(3, 100, 100)], batch_size=1, device='cpu')

def tensorboard_visualize_autoencoder():
    autoencoder = Autoencoder()
    writer = SummaryWriter(logdir='./')
    writer.add_graph(model=autoencoder, input_to_model=torch.randn(1, 3, 100, 100), verbose=False)


if __name__ == '__main__':
    tensorboard_visualize_autoencoder()
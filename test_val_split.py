import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import lightning as L

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(),nn.Linear(64, 3))

    def forward(self, x):
            return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3,64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self,x):
            return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
train_set = MNIST(os.getcwd(), download=True, train = True, transform=transforms.ToTensor())
test_set = MNIST(os.getcwd(), download=True, train = False, transform=transforms.ToTensor())

train_set_size = int(len(train_set) * 0.8 )
val_set_size = len(train_set) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(train_set, [train_set_size, val_set_size], generator=seed)   


train_loader = DataLoader(train_set)
valid_loader = DataLoader(val_set)
test_loader = DataLoader(test_set)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = L.Trainer(accelerator="gpu", devices=1, limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(autoencoder, dataloaders = test_loader)


fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = autoencoder.encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
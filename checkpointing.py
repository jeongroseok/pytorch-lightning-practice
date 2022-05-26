import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    num_items = 100

    def __getitem__(self, index):
        x = torch.tensor(float(index) / self.num_items)
        y = x**2
        return x, y

    def __len__(self):
        return self.num_items


class MyModel(LightningModule):
    def __init__(self, lr=0.01) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.layer1(x)
        y_hat = self.layer2(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(MyDataset())


def main():
    cli = LightningCLI(MyModel)
    pass


if __name__ == "__main__":
    main()

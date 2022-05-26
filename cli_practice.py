import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY
from pytorch_lightning import LightningModule

class MyDataset(Dataset):
    def __getitem__(self, index):
        return torch.tensor(index / 10.0)
    
    def __len__(self):
        return 10

class MyModel(LightningModule):
    def __init__(self, lr=0.01) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.layer = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        preds = self.layer(batch)
        loss = self.criterion(preds, batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(MyDataset())


def main():
    cli = LightningCLI()
    print(cli.config)
    print(cli.config_init)
    print(cli.datamodule)
    print(cli.datamodule_class)
    print(cli.model_class)
    pass


if __name__ == "__main__":
    main()

import argparse
import os
from pprint import pprint

import torch
import torch.nn as nn
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy
from torchvision import models


def set_persistent_workers(datamodule):
    def _data_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    datamodule._data_loader = _data_loader


class MyModel(LightningModule):
    class __HPARAMS:
        num_classes: int
        learning_rate: float

    hparams: __HPARAMS

    def __init__(self, num_classes, learning_rate=0.01) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.__build_model()

        self.criterion_ce = nn.CrossEntropyLoss()
        self.metric_accuracy = Accuracy()

    def __build_model(self):
        backbone = models.resnet18(pretrained=True)
        num_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(num_features, self.hparams.num_classes)

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        features = self.backbone(x).flatten(1)
        y_hat = self.head(features)
        return y_hat

    def __step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion_ce(y_hat, y)

        output = {"y": y, "y_hat": y_hat, "loss": loss}
        output["logs"] = {
            "accuracy": self.metric_accuracy(y_hat, y),
            "loss": loss,
        }

        return output

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in output["logs"].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return output["loss"]

    def validation_step(self, batch, batch_idx: int) -> None:
        output = self.__step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v for k, v in output["logs"].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx: int) -> None:
        output = self.__step(batch, batch_idx)
        self.log_dict(
            {f"test_{k}": v for k, v in output["logs"].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.learning_rate
        )


def main():
    seed_everything(7)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="number of gpus to use for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="maximum number of epochs for training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets",
        help="the directory to load data from",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="the learning rate to use during model training",
    )
    args = parser.parse_args()

    set_persistent_workers(CIFAR10DataModule)

    datamodule = CIFAR10DataModule(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=min(2, os.cpu_count()),
        pin_memory=True,
    )
    model = MyModel(datamodule.num_classes, args.learning_rate)

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy="ddp",
        # logger=TensorBoardLogger("lightning_logs/", name="resnet"),
        callbacks=[EarlyStopping(monitor="val_loss")],
    )

    trainer.fit(model, datamodule=datamodule)
    print("finished fitting")
    trainer.test(model, datamodule=datamodule)
    print("finished testing")
    pprint(args)


if __name__ == "__main__":
    main()

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from shark_species_classifier.models import build_model, freeze_backbone, unfreeze_all


class SharkClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        lr,
        weight_decay,
        class_weights=None,
        class_names=None,
        pretrained=True,
        freeze_backbone_epochs=0,
        cnn_dropout=0.3,
        max_epochs=10,
    ):
        super().__init__()
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        self.save_hyperparameters(
            {
                "model_name": model_name,
                "num_classes": num_classes,
                "lr": lr,
                "weight_decay": weight_decay,
                "class_names": class_names,
                "pretrained": pretrained,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "cnn_dropout": cnn_dropout,
                "max_epochs": max_epochs,
            }
        )

        self.model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            cnn_dropout=cnn_dropout,
        )

        self.freeze_backbone_epochs = freeze_backbone_epochs
        if model_name == "resnet34" and freeze_backbone_epochs > 0:
            freeze_backbone(self.model)

        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.to(dtype=torch.float32))

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_macro_f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.val_macro_f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.test_macro_f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )

    def forward(self, images):
        return self.model(images)

    def on_train_epoch_start(self):
        if (
            self.hparams["model_name"] == "resnet34"
            and self.freeze_backbone_epochs > 0
            and self.current_epoch == self.freeze_backbone_epochs
        ):
            unfreeze_all(self.model)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        self.train_acc(logits, labels)
        self.train_macro_f1(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_macro_f1",
            self.train_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        self.val_acc(logits, labels)
        self.val_macro_f1(logits, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_macro_f1",
            self.val_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        self.test_acc(logits, labels)
        self.test_macro_f1(logits, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_macro_f1", self.test_macro_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams["lr"]),
            weight_decay=float(self.hparams["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.hparams["max_epochs"])
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

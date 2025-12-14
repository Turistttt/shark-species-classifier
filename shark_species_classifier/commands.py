from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torchvision import transforms

from shark_species_classifier.data import SharkDataModule
from shark_species_classifier.lightning_module import SharkClassifier
from shark_species_classifier.utils import ensure_data_available, get_git_commit_id


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    command = str(cfg.command).lower()
    if command == "train":
        train(cfg)
    elif command == "infer":
        infer(cfg)
    else:
        raise ValueError(f"Unknown command: {cfg.command}")


def train(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    raw_dir = to_absolute_path(cfg.data.raw_dir)
    ensure_data_available(
        raw_dir=Path(raw_dir),
        yandex_public_url=cfg.data.yandex_public_url,
    )

    data_module = SharkDataModule(
        raw_dir=raw_dir,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        image_size=int(cfg.data.image_size),
        mean=list(cfg.data.mean),
        std=list(cfg.data.std),
        test_ratio=float(cfg.data.test_ratio),
        val_ratio_within_train=float(cfg.data.val_ratio_within_train),
        random_state=int(cfg.data.random_state),
    )
    data_module.setup("fit")

    model_name = str(cfg.model.name)
    num_classes = len(data_module.class_names)
    pretrained = bool(cfg.model.get("resnet34", {}).get("pretrained", True))
    freeze_epochs = int(cfg.model.get("resnet34", {}).get("freeze_backbone_epochs", 0))
    cnn_dropout = float(cfg.model.get("cnn", {}).get("dropout", 0.3))

    tracking_uri = str(cfg.mlflow.tracking_uri)
    hyperparams = {
        "seed": int(cfg.seed),
        "batch_size": int(cfg.data.batch_size),
        "num_workers": int(cfg.data.num_workers),
        "image_size": int(cfg.data.image_size),
        "model": model_name,
        "lr": float(cfg.optimizer.lr),
        "weight_decay": float(cfg.optimizer.weight_decay),
        "max_epochs": int(cfg.trainer.max_epochs),
        "freeze_backbone_epochs": freeze_epochs,
        "num_classes": num_classes,
    }

    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=str(cfg.mlflow.experiment_name),
            tracking_uri=tracking_uri,
            tags={"git_commit": get_git_commit_id()},
        )
        mlflow_logger.log_hyperparams(hyperparams)
    except Exception:
        mlflow_logger = MLFlowLogger(
            experiment_name=str(cfg.mlflow.experiment_name),
            tracking_uri="file:./mlruns",
            tags={"git_commit": get_git_commit_id()},
        )
        mlflow_logger.log_hyperparams(hyperparams)

    checkpoint_dir = to_absolute_path(cfg.paths.checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=str(cfg.trainer.checkpoint.filename),
        monitor=str(cfg.trainer.checkpoint.monitor),
        mode=str(cfg.trainer.checkpoint.mode),
        save_top_k=int(cfg.trainer.checkpoint.save_top_k),
    )
    early_stopping = EarlyStopping(
        monitor=str(cfg.trainer.early_stopping.monitor),
        mode=str(cfg.trainer.early_stopping.mode),
        patience=int(cfg.trainer.early_stopping.patience),
    )

    lightning_module = SharkClassifier(
        model_name=model_name,
        num_classes=num_classes,
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
        class_weights=data_module.class_weights
        if data_module.class_weights is not None
        else torch.ones(num_classes),
        class_names=data_module.class_names,
        pretrained=pretrained,
        freeze_backbone_epochs=freeze_epochs,
        cnn_dropout=cnn_dropout,
        max_epochs=int(cfg.trainer.max_epochs),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        fast_dev_run=bool(cfg.trainer.fast_dev_run),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        logger=[mlflow_logger],
        callbacks=[checkpoint_callback, early_stopping],
    )

    trainer.fit(lightning_module, datamodule=data_module)
    trainer.test(lightning_module, datamodule=data_module, ckpt_path="best")


def infer(cfg):
    if cfg.infer.image_path is None:
        raise ValueError("Provide infer.image_path")

    if cfg.infer.checkpoint_path is None:
        checkpoints_dir = Path(to_absolute_path(cfg.paths.checkpoints_dir))
        candidates = sorted(
            checkpoints_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .ckpt files found in '{checkpoints_dir}'. Provide infer.checkpoint_path."
            )
        checkpoint_path = str(candidates[-1])
    else:
        checkpoint_path = to_absolute_path(cfg.infer.checkpoint_path)
    image_path = to_absolute_path(cfg.infer.image_path)

    model = SharkClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((int(cfg.data.image_size), int(cfg.data.image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(cfg.data.mean), std=list(cfg.data.std)),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = logits.softmax(dim=-1).squeeze(0).cpu()

    top_k = int(cfg.infer.top_k)
    top_probs, top_indices = torch.topk(probs, k=top_k)

    class_names = list(model.hparams["class_names"])
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=False):
        print(f"{class_names[idx]}: {prob:.4f}")


if __name__ == "__main__":
    main()

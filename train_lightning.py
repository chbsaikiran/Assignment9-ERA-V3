import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import torchvision
from torch import nn
import utils
from torch.utils.data import DataLoader
import presets
from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import torchmetrics

class ImageNetLightningModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Create model
        self.num_classes = args.num_classes
        self.model = torchvision.models.get_model(
            args.model, 
            weights=args.weights, 
            num_classes=self.num_classes
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
        # Metrics
        self.train_acc1 = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1)
        self.train_acc5 = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5)
        self.val_acc1 = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1)
        self.val_acc5 = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc1', acc1, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, sync_dist=True)
        
        # Print metrics every 100 batches
        if batch_idx % 100 == 0:
            current_epoch = self.current_epoch
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(
                f"Epoch: [{current_epoch}][{batch_idx}] "
                f"Loss: {loss:.4f}  "
                f"Acc@1 {acc1:.3f}  "
                f"Acc@5 {acc5:.3f}  "
                f"LR: {current_lr:.6f}"
            )
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        
        # Update metrics
        self.val_acc1.update(outputs, targets)
        self.val_acc5.update(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_acc1', acc1, on_epoch=True, sync_dist=True)
        self.log('val_acc5', acc5, on_epoch=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        # Calculate validation metrics
        val_acc1 = self.val_acc1.compute()
        val_acc5 = self.val_acc5.compute()
        
        print(
            f"\nValidation Epoch Results: "
            f"Acc@1 {val_acc1*100:.2f}%  "  # Multiply by 100 to show percentage
            f"Acc@5 {val_acc5*100:.2f}%\n"  # Multiply by 100 to show percentage
        )
        
        # Reset metrics
        self.val_acc1.reset()
        self.val_acc5.reset()

    def configure_optimizers(self):
        # Configure parameter groups with weight decay
        parameters = utils.set_weight_decay(
            self.model,
            self.args.weight_decay,
            norm_weight_decay=self.args.norm_weight_decay,
            custom_keys_weight_decay=None,
        )
        
        # Create optimizer
        if self.args.opt.lower().startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.opt.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        
        # Create scheduler
        if self.args.lr_scheduler == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs - self.args.lr_warmup_epochs,
                eta_min=self.args.lr_min,
            )
        elif self.args.lr_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args.lr_step_size,
                gamma=self.args.lr_gamma,
            )
            
        # Add warmup scheduler if needed
        if self.args.lr_warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.args.lr_warmup_decay,
                total_iters=self.args.lr_warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.args.lr_warmup_epochs],
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    # Dataset parameters
    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    
    # Optimizer parameters
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for normalization layers")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing")
    
    # Learning rate scheduler parameters
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule")
    
    # Augmentation parameters
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability")
    
    # Model specific parameters
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method")
    parser.add_argument("--val-resize-size", default=256, type=int, help="validation resize size")
    parser.add_argument("--val-crop-size", default=224, type=int, help="validation crop size")
    parser.add_argument("--train-crop-size", default=224, type=int, help="training crop size")
    
    # Other parameters
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--amp", action="store_true", help="Use AMP training")

    # Checkpoint loading
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (e.g., output/model-12-79.15.ckpt)"
    )

    # Add the missing arguments
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        help="print frequency"
    )
    parser.add_argument(
        "--mixup-alpha",
        default=0.0,
        type=float,
        help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha",
        default=0.0,
        type=float,
        help="cutmix alpha (default: 0.0)"
    )

    return parser

def main(args):
    # Data loading code
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    
    # Create transforms
    interpolation = InterpolationMode(args.interpolation)
    train_transform = presets.ClassificationPresetTrain(
        crop_size=args.train_crop_size,
        interpolation=interpolation,
        auto_augment_policy=args.auto_augment,
        random_erase_prob=args.random_erase,
        backend=args.backend,
        use_v2=args.use_v2,
    )
    
    val_transform = presets.ClassificationPresetEval(
        crop_size=args.val_crop_size,
        resize_size=args.val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )
    
    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transform)
    
    # Set number of classes
    args.num_classes = len(train_dataset.classes)
    
    # Create data loaders with adjusted workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Create or load model
    if args.load_checkpoint:
        if args.load_checkpoint.endswith(".pth"):
            print(f"Loading model from .pth checkpoint: {args.load_checkpoint}")
            # Initialize the model according to args
            model = ImageNetLightningModel(args)
            # Load the state dictionary
            checkpoint = torch.load(args.load_checkpoint)
            model.load_state_dict(checkpoint)
        else:
            print(f"Loading model from .ckpt checkpoint: {args.load_checkpoint}")
            model = ImageNetLightningModel.load_from_checkpoint(args.load_checkpoint, args=args)
    else:
        print("Creating new model")
        model = ImageNetLightningModel(args)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='model-{epoch:02d}-{val_acc1:.2f}',
            monitor='val_acc1',
            mode='max',
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=-1,  # Use all available GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16 if args.amp else 32,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    # Adjust number of workers to avoid warning
    if args.workers > 2:
        print(f"Reducing number of workers from {args.workers} to 2")
        args.workers = 2
        
    main(args) 
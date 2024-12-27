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
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_acc1', acc1, on_epoch=True, sync_dist=True)
        self.log('val_acc5', acc5, on_epoch=True, sync_dist=True)
        
        # Print metrics for the first batch of validation
        if batch_idx == 0:
            print(
                f"\nValidation: "
                f"Loss: {loss:.4f}  "
                f"Acc@1 {acc1:.3f}  "
                f"Acc@5 {acc5:.3f}"
            )
        
        return loss

    def validation_epoch_end(self, outputs):
        # This is called at the end of validation
        val_loss = torch.stack([x for x in outputs]).mean()
        val_acc1 = self.val_acc1.compute()
        val_acc5 = self.val_acc5.compute()
        
        print(
            f"\nValidation Epoch End: "
            f"Loss: {val_loss:.4f}  "
            f"Acc@1 {val_acc1:.3f}  "
            f"Acc@5 {val_acc5:.3f}\n"
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
    
    # Create data loaders
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
    
    # Create model
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
    from train import get_args_parser
    args = get_args_parser().parse_args()
    main(args) 
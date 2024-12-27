#!/bin/bash

aws s3 cp s3://resnet50dataset/imagenet-object-localization-challenge.zip /home/ubuntu/Repos/Assignment9/data/imagenet-object-localization-challenge.zip
cd data
unzip imagenet-object-localization-challenge.zip
cd ..
python convert_val.py -d ./data/ILSVRC/Data/CLS-LOC/val -l ./data/LOC_val_solution.csv
python train_lightning.py --model resnet50 --data-path ./data/ILSVRC/Data/CLS-LOC --batch-size 256 --epochs 100 --lr 0.1 --opt sgd --weight-decay 1e-4 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --output-dir ./output --amp --mixup-alpha 0.2 --cutmix-alpha 1.0 --auto-augment ra --label-smoothing 0.1 --load-checkpoint ./output/model_start.pth | tee log.txt


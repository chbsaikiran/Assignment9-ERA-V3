# How To Run For CIFAR10 Dataset
1. python download_cifar.py
2. python train.py --model resnet50 --device cpu --epochs 2 --data-path ./data

python train.py --model resnet50 --data-path ./data/ILSVRC/Data/CLS-LOC --batch-size 256 --epochs 100 --print-freq 100 --lr 0.1 --opt sgd --weight-decay 1e-4 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --output-dir ./output --amp --mixup-alpha 0.2 --cutmix-alpha 1.0 --auto-augment ra --label-smoothing 0.1

# STEPS AFTER EC2 INSTANCE IS ACCESSED FROM VS CODE:
1. Ubuntu : mkdir Repos
2. Windows: scp -i F:\Saikiran\TSAI\Session9\EC2_Test\resnet50-key.pem -r F:\Saikiran\TSAI\Session9\venkatesh_saikiran_repos\Assignment9 ubuntu@ec2-15-206-148-134.ap-south-1.compute.amazonaws.com:/home/ubuntu/Repos/
3. after cd to /home/ubuntu/Repos/Assignment9 on Ubuntu : mkdir data
4. aws configure
5. Ubuntu : dos2unix copy_and_unzip.sh
6. Ubuntu: chmod +x copy_and_unzip.sh
7. Ubuntu: ./copy_and_unzip.sh
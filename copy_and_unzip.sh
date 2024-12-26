#!/bin/bash

aws s3 cp s3://resnet50dataset/imagenet-object-localization-challenge.zip /home/ubuntu/Repos/Assignment9/data/imagenet-object-localization-challenge.zip
cd data
unzip imagenet-object-localization-challenge.zip


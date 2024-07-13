""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CHINESE_TRAIN_MEAN = (51.035623246697746, 51.035623246697746, 51.035623246697746)
CHINESE_TRAIN_STD = (98.08926644129016, 98.08926644129016, 98.08926644129016)

#CHINESE_TEST_MEAN = (50.62961222180359, 50.62961222180359, 50.62961222180359)
#CHINESE_TEST_STD = (97.78069389667155, 97.78069389667155, 97.78069389667155)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 40
MILESTONES = [10, 20, 30, 40, 50, 60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 1









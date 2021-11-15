# Main file to train/test/validation

import csv
import argparse as ap
import random

import torch
import torch.nn.functional as F
import numpy as np

from data_io import load
from models import CNN1, CNN2

def get_args():
    """ Define command line arguments. """
    p = ap.ArgumentParser()

    # Mode to run the model in.
    p.add_argument("mode", choices=["train", "predict"], type=str)

    # File locations
    p.add_argument("--data-dir", type=str, default="data") # TODO: Edit default data drive
    p.add_argument("--log-file", type=str, default="simple-cnn-logs.csv") # TODO: Edit default log file
    p.add_argument("--model-save", type=str, default="simple-cnn-model.torch") # TODO: Edit default model name
    p.add_argument("--predictions-file", type=str, default="simple-cnn-preds.txt") # TODO: Edit default prediction file name

    # hyperparameters
    p.add_argument("--model", type=str, default="cnn1")
    p.add_argument("--train-steps", type=int, default=3500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.001)

    # cnn1 hparams
    p.add_argument('--cnn-n1-channels', type=int, default=80)
    p.add_argument('--cnn-n1-kernel', type=int, default=10)
    p.add_argument('--cnn-n2-kernel', type=int, default=5)

    # cnn2 hparams
    p.add_argument('--best-n1-channels', type=int, default=80)
    p.add_argument('--best-n1-kernel', type=int, default=5)
    p.add_argument('--best-n2-channels', type=int, default=60)
    p.add_argument('--best-n2-kernel', type=int, default=5)
    p.add_argument('--best-pool1', type=int, default=2)
    p.add_argument('--best-n3-channels', type=int, default=40)
    p.add_argument('--best-n3-kernel', type=int, default=3)
    p.add_argument('--best-n4-channels', type=int, default=20)
    p.add_argument('--best-n4-kernel', type=int, default=3)
    p.add_argument('--best-pool2', type=int, default=3)
    p.add_argument('--best-linear-features', type=int, default=80)
    return p.parse_args()
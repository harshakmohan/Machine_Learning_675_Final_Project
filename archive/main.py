# Main file to train/test/validation

import csv
import argparse as ap

import torch
import gc
import torch.nn.functional as F
import numpy as np
gc.collect()
torch.cuda.empty_cache()

from archive.data_io import DataIO
from models import CNN1, CNN2

def get_args():
    """ Define command line arguments. """
    p = ap.ArgumentParser()

    # Mode to run the model in.
    p.add_argument("mode", choices=["train", "predict"], type=str)

    # File locations
    p.add_argument("--data-dir", type=str, default="data") # TODO: Edit default data drive
    p.add_argument("--log-file", type=str, default="cnn1.csv") # TODO: Edit default log file
    p.add_argument("--model-save", type=str, default="cnn1.torch") # TODO: Edit default model name
    p.add_argument("--predictions-file", type=str, default="cnn1-preds.txt") # TODO: Edit default prediction file name

    # hyperparameters
    p.add_argument("--model", type=str, default="cnn1")
    p.add_argument("--train-steps", type=int, default=3500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)

    # cnn1 hparams
    p.add_argument('--cnn-n1-channels', type=int, default=80)
    p.add_argument('--cnn-n1-kernel', type=int, default=10)
    p.add_argument('--cnn-n2-kernel', type=int, default=5)

    # cnn2 hparams
    # p.add_argument('--best-n1-channels', type=int, default=80)
    # p.add_argument('--best-n1-kernel', type=int, default=5)
    # p.add_argument('--best-n2-channels', type=int, default=60)
    # p.add_argument('--best-n2-kernel', type=int, default=5)
    # p.add_argument('--best-pool1', type=int, default=2)
    # p.add_argument('--best-n3-channels', type=int, default=40)
    # p.add_argument('--best-n3-kernel', type=int, default=3)
    # p.add_argument('--best-n4-channels', type=int, default=20)
    # p.add_argument('--best-n4-kernel', type=int, default=3)
    # p.add_argument('--best-pool2', type=int, default=3)
    # p.add_argument('--best-linear-features', type=int, default=80)
    return p.parse_args()


def train(args):
    # Set Random Seed for Consistency Across Runs
    #RAND_SEED = 42
    #torch.manual_seed(RAND_SEED)
    #np.random.seed(RAND_SEED)
    #random.seed(42)
    # setup metric logging. It's important to log your loss!!
    log_f = open(args.log_file, 'w')
    fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
    logger = csv.DictWriter(log_f, fieldnames)
    logger.writeheader()

    # prepare gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    io = DataIO()
    io.create_dataset()  # creates dataset (.npy files for X and y)

    train_data = np.load('X.npy', allow_pickle=True)
    #train_data = torch.from_numpy(train_data).to(device)

    train_labels = np.load('y.npy', allow_pickle=True)
    #train_labels = torch.from_numpy(train_labels).to(device)
    #train_data, train_labels = load(args.data_dir, split="train")  #TODO: Update with create_dataset
    #dev_data, dev_labels = load(args.data_dir, split="dev")  #TODO: Update with create_dataset

    # Build model
    if args.model.lower() == "cnn1":
        model = CNN1(img_size=args.img_size).to(device)
        print('model state: ', model.conv1.weight.device)
    elif args.model.lower() == "cnn2":
        model = CNN2(args.cnn_n1_channels,
                            args.cnn_n1_kernel,
                            args.cnn_n2_kernel).to(device)
        print('model state: ', model.conv1.weight.device)

    # elif args.model.lower() == "cnn3":
    #     model = BestNN(args.best_n1_channels,
    #                    args.best_n1_kernel,
    #                    args.best_n2_channels,
    #                    args.best_n2_kernel,
    #                    args.best_pool1,
    #                    args.best_n3_channels,
    #                    args.best_n3_kernel,
    #                    args.best_n4_channels,
    #                    args.best_n4_kernel,
    #                    args.best_pool2,
    #                    args.best_linear_features1,
    #                    args.best_linear_features2,
    #                    args.best_dropout).to(device)
    else:
        raise Exception("Unknown model type passed in!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Can change this loop as you need to, to optimize your training!
    # for example, if you wanted to implement early stopping to make sure you
    # don't overfit your model, you would do so in this loop.
    best_model = None
    best_dev_acc = 0.0
    for step in range(args.train_steps):
        # run the model and backprop for train steps

        i = np.random.choice(train_data.shape[0], size=args.batch_size, replace=True) # TODO: Change back to replace=False for real training
        x = torch.from_numpy(train_data[i].astype(np.float32)).to(device)
        y = torch.from_numpy(train_labels[i].astype(np.int64)).to(device)
        print('x device: ', x.is_cuda)
        print('y device: ', y.is_cuda)

        # Forward pass: Get logits for x
        logits = model(x).squeeze() # removing .squeeze() for best neural network
        #print('logits type: ', logits.is_cuda)
        print('logits type: ', type(logits))
        print('logits size: ', logits.size())
        print('y type: ', type(y))
        print('y size: ', y.size())


        # Compute loss
        #pdb.set_trace()
        loss = F.cross_entropy(logits, y.long())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every 100 steps, log metrics
        if step % 100 == 0:
            train_acc, train_loss = approx_train_acc_and_loss(model,
                                                              train_data,
                                                              train_labels)
            dev_acc, dev_loss = dev_acc_and_loss(model, train_data, train_labels) # changed it to train_data and train_labels

            if dev_acc > best_dev_acc:
                best_model = model
                best_dev_acc = dev_acc

            step_metrics = {
                'step': step,
                'train_loss': loss.item(),
                'train_acc': train_acc,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc
            }

            print(f'On step {step}: Train loss {train_loss} | Dev acc is {dev_acc}')
            logger.writerow(step_metrics)

    # close the log file
    log_f.close()
    # save model
    print(f'Done training. Saving model at {args.model_save}')
    torch.save(best_model, args.model_save)


def approx_train_acc_and_loss(model, train_data, train_labels):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    idxs = np.random.choice(len(train_data), 4000, replace=True)  # TODO: change back to replace = False for training
    x = torch.from_numpy(train_data[idxs].astype(np.float32)).to(device)
    y = torch.from_numpy(train_labels[idxs].astype(np.int)).to(device)
    logits = model(x).squeeze()
    loss = F.cross_entropy(logits, y.long()) # changed from y to y.long() for best nn
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.cpu().numpy()), loss.item()


def dev_acc_and_loss(model, dev_data, dev_labels):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.from_numpy(dev_data.astype(np.float32)).to(device)
    y = torch.from_numpy(dev_labels.astype(np.int)).to(device)
    logits = model(x).squeeze()
    loss = F.cross_entropy(logits, y.long()) # changed from y to y.long() for best nn
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred.cpu().numpy()), loss.item()


def accuracy(y, y_hat):
    return (y == y_hat).astype(np.float).mean()


def test(args):
    # You should not change this function at all
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # added this line
    model = torch.load(args.model_save).to(device)  # modified this line
    test_data, _ = load(args.data_dir, split="test", load_labels=False)

    preds = []
    for test_ex in test_data:
        x = torch.from_numpy(test_ex.astype(np.float32)).to(device)  # modified this line
        # Make the x look like it's in a batch of size 1
        x = x.view(1, -1)
        model.eval()
        logits = model(x)
        pred = torch.max(logits, 1)[1]
        preds.append(pred.item())
    print(f'Done making predictions! Storing them in {args.predictions_file}')
    preds = np.array(preds)
    np.savetxt(args.predictions_file, preds, fmt='%d')

if __name__ == '__main__':
    ARGS = get_args()
    if ARGS.mode == 'train':
        train(ARGS)
    elif ARGS.mode == 'predict':
        test(ARGS)
    else:
        print(f'Invalid mode: {ARGS.mode}! Must be either "train" or "predict".')
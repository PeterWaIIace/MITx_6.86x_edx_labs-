import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        print("input_dimension:",input_dimension,"\n")
        # TODO initialize model layers here
        self.Conv1 = nn.Conv2d(1, 8, (3, 3))
        self.ReLU1 = nn.ReLU()
        self.MaxPool2d1 = nn.MaxPool2d((2, 2))
        self.Conv2 = nn.Conv2d(8,16,(3,3))
        self.ReLU2 = nn.ReLU()
        self.MaxPool2d2 = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.Linear1 = nn.Linear(720,128)
        self.Drop1 = nn.Dropout2d(0.5)
        self.Linear_fd = nn.Linear(128,10)
        self.Linear_sd = nn.Linear(128,10)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x1 = self.MaxPool2d1(self.ReLU1(self.Conv1(x)))
        x2 = self.MaxPool2d2(self.ReLU2(self.Conv2(x1)))
        xf = self.flatten(x2)
        x3 = self.Linear1(xf)
        xD= self.Drop1(x3)

        return self.Linear_fd(xD), self.Linear_sd(xD)

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model,lr=0.1,momentum=0.15)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()

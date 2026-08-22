import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        device = device or ndl.cpu()
        dtype = dtype or "float32"

        def conv_bn(c_in, c_out, kernel_size, stride):
            return nn.Sequential(
                nn.Conv(
                    c_in,
                    c_out,
                    kernel_size,
                    stride,
                    device=device,
                    dtype=dtype,
                ),
                nn.BatchNorm2d(c_out, device=device, dtype=dtype),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            conv_bn(3, 16, 7, 4),
            conv_bn(16, 32, 3, 2),
            nn.Residual(
                nn.Sequential(
                    conv_bn(32, 32, 3, 1),
                    conv_bn(32, 32, 3, 1),
                )
            ),
            conv_bn(32, 64, 3, 2),
            conv_bn(64, 128, 3, 2),
            nn.Residual(
                nn.Sequential(
                    conv_bn(128, 128, 3, 1),
                    conv_bn(128, 128, 3, 1),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.net(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        self.seq_model_type = seq_model
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embedded = self.embedding(x)
        sequence_output, final_hidden = self.seq_model(embedded, h)
        sequence_output = ndl.ops.reshape(
            sequence_output,
            (sequence_output.shape[0] * sequence_output.shape[1], sequence_output.shape[2]),
        )
        out = self.linear(sequence_output)
        return out, final_hidden
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 3, 32, 32), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)

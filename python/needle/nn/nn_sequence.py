"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # sigmoid(x) = 1 / (1 + exp(-x)).  Use existing differentiable
        # Needle operators rather than indexing or backend-specific code.
        denominator = ops.add_scalar(ops.exp(-x), 1)
        return ops.power_scalar(denominator, -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-1 / np.sqrt(hidden_size),
                high=1 / np.sqrt(hidden_size),
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-1 / np.sqrt(hidden_size),
                high=1 / np.sqrt(hidden_size),
                device=device,
                dtype=dtype,
            )
        )
        self.bias_ih = (
            Parameter(
                init.rand(
                    hidden_size,
                    low=-1 / np.sqrt(hidden_size),
                    high=1 / np.sqrt(hidden_size),
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )
        self.bias_hh = (
            Parameter(
                init.rand(
                    hidden_size,
                    low=-1 / np.sqrt(hidden_size),
                    high=1 / np.sqrt(hidden_size),
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(
                X.shape[0],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_next = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.bias:
            bias_ih = ops.broadcast_to(
                ops.reshape(self.bias_ih, (1, self.hidden_size)), h_next.shape
            )
            bias_hh = ops.broadcast_to(
                ops.reshape(self.bias_hh, (1, self.hidden_size)), h_next.shape
            )
            h_next = h_next + bias_ih + bias_hh
        if self.nonlinearity == "relu":
            h_next = ops.relu(h_next)
        else:
            h_next = ops.tanh(h_next)
        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size
            self.rnn_cells.append(
                RNNCell(
                    cell_input_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype,
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_n = []
        time_steps = ops.split(X, axis=0)
        h0_layers = ops.split(h0, axis=0)
        for layer in range(self.num_layers):
            h = h0_layers[layer]
            layer_output = []
            for t in range(seq_len):
                h = self.rnn_cells[layer](time_steps[t], h)
                layer_output.append(h)
            time_steps = layer_output
            h_n.append(h)
        h_n = ops.stack(h_n, axis=0)
        return ops.stack(time_steps, axis=0), h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        gate_size = 4 * hidden_size

        self.W_ih = Parameter(
            init.rand(
                input_size,
                gate_size,
                low=-1 / np.sqrt(hidden_size),
                high=1 / np.sqrt(hidden_size),
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                gate_size,
                low=-1 / np.sqrt(hidden_size),
                high=1 / np.sqrt(hidden_size),
                device=device,
                dtype=dtype,
            )
        )
        self.bias_ih = (
            Parameter(
                init.rand(
                    gate_size,
                    low=-1 / np.sqrt(hidden_size),
                    high=1 / np.sqrt(hidden_size),
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )
        self.bias_hh = (
            Parameter(
                init.rand(
                    gate_size,
                    low=-1 / np.sqrt(hidden_size),
                    high=1 / np.sqrt(hidden_size),
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h_prev = init.zeros(
                X.shape[0],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
            c_prev = init.zeros(
                X.shape[0],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        else:
            h_prev, c_prev = h

        gates = X @ self.W_ih + h_prev @ self.W_hh
        if self.bias:
            bias_ih = ops.broadcast_to(
                ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)),
                gates.shape,
            )
            bias_hh = ops.broadcast_to(
                ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)),
                gates.shape,
            )
            gates = gates + bias_ih + bias_hh

        # reshape gates from (bs, 4*hidden_size) to (bs, 4, hidden_size) 
        gates = ops.reshape(gates, (X.shape[0], 4, self.hidden_size))
        gates = ops.split(gates, axis=1)
        i = self.sigmoid(gates[0])
        f = self.sigmoid(gates[1])
        g = ops.tanh(gates[2])
        o = self.sigmoid(gates[3])

        c_next = f * c_prev + i * g
        h_next = o * ops.tanh(c_next)
        return h_next, c_next
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(
                LSTMCell(
                    cell_input_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            h0 = init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
            c0 = init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        else:
            h0, c0 = h

        h0_layers = ops.split(h0, axis=0)
        c0_layers = ops.split(c0, axis=0)
        time_steps = ops.split(X, axis=0)
        h_n = []
        c_n = []

        for layer in range(self.num_layers):
            h = h0_layers[layer]
            c = c0_layers[layer]
            layer_output = []
            for t in range(seq_len):
                h, c = self.lstm_cells[layer](time_steps[t], (h, c))
                layer_output.append(h)
            time_steps = layer_output
            h_n.append(h)
            c_n.append(c)

        return ops.stack(time_steps, axis=0), (
            ops.stack(h_n, axis=0),
            ops.stack(c_n, axis=0),
        )
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_one_hot = init.one_hot(
            self.num_embeddings,
            x,
            device=x.device,
            dtype=self.dtype,
        )
        x_one_hot = ops.reshape(x_one_hot, (seq_len * bs, self.num_embeddings))
        output = x_one_hot @ self.weight
        return ops.reshape(output, (seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION

import LibCall
from .module import Module
from .... import torch
from .. import functional as F


class RNNBase(Module):
    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        **kwargs
    ):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1


class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__("RNN_TANH", *args, **kwargs)

    def forward(self, value):
        # TODO: resolve packed sequence
        input, h_0 = value

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        assert LibCall.guard.require_eq(h_0.ndim, 3, "h_0 rank is not 3")

        input_shape = input.shape
        h_0_shape = h_0.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.rand(
                batch, self.num_layers * self.num_directions, self.hidden_size
            )
            output_shape = (batch, seq_len, self.num_directions * self.hidden_size)
            h_n_shape = (
                batch,
                self.num_layers * self.num_directions,
                self.hidden_size,
            )
        else:
            batch = input_shape[1]
            seq_len = input_shape[0]
            h_0_require = torch.rand(
                self.num_layers * self.num_directions, batch, self.hidden_size
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        assert LibCall.guard.require_shape_eq(
            h_0_shape, h_0_require, "h_0 shape mismatch"
        )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)

        return (output, h_n)


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__("LSTM", *args, **kwargs)

    def forward(self, value):
        # TODO: resolve packed sequence
        input, h_0 = value

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        assert LibCall.guard.require_eq(h_0.ndim, 3, "h_0 rank is not 3")

        input_shape = input.shape
        h_0_shape = h_0.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.rand(
                batch, self.num_layers * self.num_directions, self.hidden_size
            )
            output_shape = (batch, seq_len, self.num_directions * self.hidden_size)
            h_n_shape = (
                batch,
                self.num_layers * self.num_directions,
                self.hidden_size,
            )
        else:
            batch = input_shape[1]
            seq_len = input_shape[0]
            h_0_require = torch.rand(
                self.num_layers * self.num_directions, batch, self.hidden_size
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        assert LibCall.guard.require_shape_eq(
            h_0_shape, h_0_require, "h_0 shape mismatch"
        )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)

        return (output, h_n)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__("GRU", *args, **kwargs)

    def forward(self, value):
        # TODO: resolve packed sequence
        input, h_0 = value

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        assert LibCall.guard.require_eq(h_0.ndim, 3, "h_0 rank is not 3")

        input_shape = input.shape
        h_0_shape = h_0.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.rand(
                batch, self.num_layers * self.num_directions, self.hidden_size
            )
            output_shape = (batch, seq_len, self.num_directions * self.hidden_size)
            h_n_shape = (
                batch,
                self.num_layers * self.num_directions,
                self.hidden_size,
            )
        else:
            batch = input_shape[1]
            seq_len = input_shape[0]
            h_0_require = torch.rand(
                self.num_layers * self.num_directions, batch, self.hidden_size
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        assert LibCall.guard.require_shape_eq(
            h_0_shape, h_0_require, "h_0 shape mismatch"
        )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)

        return (output, h_n)


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

    def flatten_parameters(self):
        return


class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__("RNN_TANH", *args, **kwargs)

    def forward(self, input, hx=None):
        # TODO: resolve packed sequence

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        input_shape = input.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.Size(
                (batch, self.num_layers * self.num_directions, self.hidden_size)
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
            h_0_require = torch.Size(
                (self.num_layers * self.num_directions, batch, self.hidden_size)
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        if hx is not None:
            assert LibCall.guard.require_shape_eq(
                hx.shape, h_0_require, "h_0 shape mismatch"
            )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)

        return (output, h_n)


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__("LSTM", *args, **kwargs)

    def forward(self, input, hx=None):
        # TODO: resolve packed sequence

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        input_shape = input.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.Size(
                (batch, self.num_layers * self.num_directions, self.hidden_size)
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
            h_0_require = torch.Size(
                (self.num_layers * self.num_directions, batch, self.hidden_size)
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        if hx is not None:
            h_0, c_0 = hx
            assert LibCall.guard.require_shape_eq(
                h_0.shape, h_0_require, "h_0 shape mismatch"
            )
            assert LibCall.guard.require_shape_eq(
                c_0.shape, h_0_require, "c_0 shape mismatch"
            )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)
        c_n = torch.rand(*h_n_shape)

        return (output, (h_n, c_n))


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__("GRU", *args, **kwargs)

    def forward(self, input, hx=None):
        # TODO: resolve packed sequence

        assert LibCall.guard.require_eq(input.ndim, 3, "input rank is not 3")
        input_shape = input.shape

        if self.batch_first:
            batch = input_shape[0]
            seq_len = input_shape[1]
            h_0_require = torch.Size(
                (batch, self.num_layers * self.num_directions, self.hidden_size)
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
            h_0_require = torch.Size(
                (self.num_layers * self.num_directions, batch, self.hidden_size)
            )
            output_shape = (seq_len, batch, self.num_directions * self.hidden_size)
            h_n_shape = (
                self.num_layers * self.num_directions,
                batch,
                self.hidden_size,
            )

        if hx is not None:
            assert LibCall.guard.require_shape_eq(
                hx.shape, h_0_require, "h_0 shape mismatch"
            )

        output = torch.rand(*output_shape)
        h_n = torch.rand(*h_n_shape)

        return (output, h_n)


class RNNCellBase(Module):
    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = torch.Tensor(num_chunks * hidden_size, input_size)
        self.weight_hh = torch.Tensor(num_chunks * hidden_size, hidden_size)
        if bias:
            self.bias_ih = torch.Tensor(num_chunks * hidden_size)
            self.bias_hh = torch.Tensor(num_chunks * hidden_size)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def check_forward_input(self, input) -> None:
        assert LibCall.guard.require_eq(
            input.shape[1], self.input_size, "input has inconsistent input_size"
        )

    def check_forward_hidden(self, input, hx, hidden_label=""):
        assert LibCall.guard.require_eq(
            input.shape[0],
            hx.shape[0],
            "Input batch size doesn't match hidden batch size",
        )
        assert LibCall.guard.require_eq(
            hx.shape[1], self.hidden_size, "hidden has inconsistent hidden_size"
        )

    def reset_parameters(self):
        pass


class RNNCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is not None:
            self.check_forward_hidden(input, hx)
            return LibCall.torch.identityShape(hx)
        else:
            return torch.rand(input.size(0), self.hidden_size, dtype=input.dtype)


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is not None:
            self.check_forward_hidden(input, hx[0])
            self.check_forward_hidden(input, hx[1])
            return (
                LibCall.torch.identityShape(hx[0]),
                LibCall.torch.identityShape(hx[1]),
            )
        else:
            return (
                torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype),
                torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype),
            )


class GRUCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is not None:
            self.check_forward_hidden(input, hx)
            return LibCall.torch.identityShape(hx)
        else:
            return torch.rand(input.size(0), self.hidden_size, dtype=input.dtype)

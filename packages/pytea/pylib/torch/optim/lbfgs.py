from .optimizer import Optimizer


class LBFGS(Optimizer):
    def __init__(
        self,
        params,
        lr=1,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
    ):
        super(LBFGS, self).__init__(params)

    def step(self, closure):
        closure()

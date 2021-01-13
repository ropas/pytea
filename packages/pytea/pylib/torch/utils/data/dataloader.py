import LibCall
from ...tensor import Tensor


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._last_batch = 0

        self.datalen = len(self.dataset)
        self._len = None

    def __len__(self):
        if self._len is not None:
            return self._len

        if self.drop_last == False and self.datalen % self.batch_size > 0:
            self._last_batch = self.datalen % self.batch_size
            self._len = self.datalen // self.batch_size + 1
        else:
            self._len = self.datalen // self.batch_size

        return self._len

    def __getitem__(self, index):
        item, target = self.dataset[index]
        _len = len(self)

        if self.drop_last == False and self._last_batch > 0 and index == _len - 1:
            batch_size = self._last_batch
        else:
            batch_size = self.batch_size

        if isinstance(item, list) or isinstance(item, tuple):
            ret_list = []
            for list_item in item:
                if isinstance(list_item, Tensor):
                    ret_list.append(LibCall.shape.repeat(list_item, 0, batch_size))
                else:
                    ret_list.append(Tensor(batch_size))
            return ret_list, Tensor([batch_size])
        else:
            if isinstance(item, Tensor):
                return LibCall.shape.repeat(item, 0, batch_size), Tensor([batch_size])
            else:
                return Tensor(batch_size)), Tensor([batch_size])

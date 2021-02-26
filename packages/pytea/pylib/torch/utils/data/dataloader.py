import LibCall
from ...tensor import Tensor
from PIL import Image


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
        item_tuple = self.dataset[index * self.batch_size]
        _len = len(self)

        if self.drop_last == False and self._last_batch > 0 and index == _len - 1:
            batch_size = self._last_batch
        # TODO:
        # elif _len == 0:
        else:
            batch_size = self.batch_size

        ret_list = []
        for item in item_tuple:
            if isinstance(item, list) or isinstance(item, tuple):
                ret_item = []
                for list_item in item:
                    if isinstance(list_item, Tensor) and list_item.dim() > 0:
                        ret_item.append(LibCall.shape.repeat(list_item, 0, batch_size))
                    else:
                        ret_item.append(Tensor(batch_size))
                ret_list.append(ret_item)
            else:
                if isinstance(item, Tensor) and item.dim() > 0:
                    ret_list.append(LibCall.shape.repeat(item, 0, batch_size))
                else:
                    ret_list.append(Tensor(batch_size))

        if len(ret_list) == 1:
            return ret_list[0]
        else:
            return ret_list

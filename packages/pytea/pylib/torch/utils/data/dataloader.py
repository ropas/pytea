import LibCall
import math
import torch


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

        datalen = len(dataset)
        if self.drop_last == True:
            self._len = datalen // self.batch_size
            self._last_batch = batch_size
        else:
            self._last_batch = (datalen - 1) % self.batch_size
            remainder = math.ceil(self._last_batch / self.batch_size)
            self._last_batch += 1
            self._len = datalen // self.batch_size + remainder

        self._len = LibCall.guard.new_symbol_int("DataLoader_Len", self._len)

        self._curr_idx = 0

    def __len__(self):
        # box constant value to prevent constant iteration by for-loop
        # to prevent boxing, set 'boxDataLoader' option to false in 'pyteaconfig.json'
        return LibCall.builtins.box(self._len)

    def __getitem__(self, index):
        item_tuple = self.dataset[index * self.batch_size]

        if self.drop_last == True:
            batch_size = self.batch_size
        else:
            # if index == self._len - 1:
            #     batch_size = self._last_batch
            # else:
            #     batch_size = self.batch_size
            # below is single expression version of above
            lb = self.batch_size - self._last_batch
            li = self._len - 1
            step = 1 // (abs(index - li) + 1)
            batch_size = self.batch_size - lb * step

        # return new symbolic variable if batch_size is not a constant
        batch_size = LibCall.guard.new_symbol_int("DataLoader_Batch", batch_size)

        ret_list = []
        for item in item_tuple:
            if isinstance(item, list) or isinstance(item, tuple):
                ret_item = []
                for list_item in item:
                    if isinstance(list_item, torch.Tensor) and list_item.dim() > 0:
                        ret_item.append(
                            torch.Tensor(
                                LibCall.shape.repeat(list_item.shape, 0, batch_size)
                            )
                        )
                    else:
                        ret_item.append(torch.Tensor(batch_size))
                ret_list.append(ret_item)
            else:
                if isinstance(item, torch.Tensor) and item.dim() > 0:
                    ret_list.append(
                        torch.Tensor(LibCall.shape.repeat(item.shape, 0, batch_size))
                    )
                else:
                    ret_list.append(torch.Tensor(batch_size))

        if len(ret_list) == 1:
            return ret_list[0]
        else:
            return ret_list

    def __iter__(self):
        # TODO: precise iteration
        return self

    def __next__(self):
        item = self[self._curr_idx]
        self._curr_idx += 1
        return item

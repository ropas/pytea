import LibCall
import torch
import torch.utils.data as data


class DataLabelList:
    def __init__(self, baseDataset, _len):
        self.baseDataset = baseDataset
        self._len = _len

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise IndexError("Dataset out of bound")

        return self.baseDataset[index]


# TODO: implement this for general array-like values
def train_test_split(*arrays, **options):
    ret_arrays = []

    # test_size = options.pop("test_size", None)
    # train_size = options.pop("train_size", None)

    test_size = options["test_size"]
    train_size = None

    if test_size is None and train_size is None:
        raise RuntimeError("Either test_size or train_size should be known")

    if test_size is None:
        test_size = 1 - train_size

    # when arrays = dataset(like MNIST)
    if isinstance(arrays[0], data.Dataset):
        baseDataset = arrays[0]
        dlen = len(baseDataset)  # data length
        test_length = int(test_size * dlen)
        train_length = dlen - test_length

        trainList = DataLabelList(arrays[0], train_length)
        testList = DataLabelList(arrays[0], test_length)

        return trainList, testList

    # when arrays = (data_batch, label_batch)
    for array in arrays:
        test_length = int(test_size * len(array))
        train_length = len(array) - test_length

        train = LibCall.shape.repeat(array.shape, 0, train_length)
        train = LibCall.torch.reduce(torch.Tensor(train), 1, False)
        test = LibCall.shape.repeat(array.shape, 0, test_length)
        test = LibCall.torch.reduce(torch.Tensor(test), 1, False)

        ret_arrays.append(train)
        ret_arrays.append(test)

    return ret_arrays

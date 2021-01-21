import LibCall
import torch
import torch.utils.data.dataset as dataset


# TODO: implement this
def train_test_split(*arrays, **options):
    ret_arrays = []

    #test_size = options.pop("test_size", None)
    #train_size = options.pop("train_size", None)

    test_size = options['test_size']
    train_size = None

    if test_size is None and train_size is None:
        raise RuntimeError("Either test_size or train_size should be known")

    if test_size is None:
        test_size = 1 - train_size
    
    for array in arrays:
        test_length = int(test_size * len(array))
        train_length = len(array) - test_length

        train = LibCall.shape.repeat(array, 0, train_length)
        train = LibCall.torch.reduce(train, 1, False)
        test = LibCall.shape.repeat(array, 0, test_length)
        test = LibCall.torch.reduce(test, 1, False)

        ret_arrays.append(train)
        ret_arrays.append(test)
    
    return ret_arrays
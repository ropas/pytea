import LibCall


class Dataset:
    pass


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class _SeqSubset(Dataset):
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length

    def __getitem__(self, idx):
        return self.dataset[self.offset + idx]

    def __len__(self):
        return self.length


def random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    result = []
    for l in lengths:
        result.append(_SeqSubset(dataset, 0, l))

    return result

def exists(path):
    # TODO: check path
    return True


def join(path, *paths):
    for p in paths:
        path = path + "/" + p

    return path

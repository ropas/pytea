def legacy_get_string(size_average, reduce, emit_warning=True):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'

    return ret
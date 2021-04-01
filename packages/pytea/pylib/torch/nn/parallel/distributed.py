def DistributedDataParallel(
    module,
    device_ids=None,
    output_device=None,
    dim=0,
    broadcast_buffers=True,
    process_group=None,
    bucket_cap_mb=25,
    find_unused_parameters=False,
    check_reduction=False,
    gradient_as_bucket_view=False,
):
    return module

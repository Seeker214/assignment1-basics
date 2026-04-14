import cProfile
import pstats
import time
import io
import logging

def profile_fun(func, *args, top_n=20, sort_by="cumulative", **kwargs):
    pr = cProfile.Profile()
    wall_start = time.time()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    wall_time = time.time() - wall_start

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)
    stats_str = stream.getvalue()

    # logging.info(f"\n{'='*60}")
    # logging.info(f"函数名:      {func.__qualname__}")
    # logging.info(f"实际耗时:    {wall_time:.4f}s")
    # logging.info(f"排序方式:    {sort_by}")
    # logging.info(f"{'='*60}")
    logging.info(stats_str)
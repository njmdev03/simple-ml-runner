import time
from profiling.profiler import Profiler

def test_profiler_basic():
    p = Profiler()
    p.start("test")
    time.sleep(0.1)
    p.stop("test")

    assert "test" in p.times
    assert p.times["test"] >= 0.1

def test_profiler_accumulation():
    p = Profiler()
    p.start("test")
    time.sleep(0.05)
    p.stop("test")

    initial = p.times["test"]

    p.start("test")
    time.sleep(0.05)
    p.stop("test")

    assert p.times["test"] > initial

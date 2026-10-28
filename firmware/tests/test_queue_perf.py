# test_queue_perf.py - descriptor queue and performance counters through the driver (register model).
import random

import pytest

from tinyml_fpga import HwEngine, SimBus
from tinyml_fpga.engine import MODE_DW, MODE_RAW
from tinyml_fpga.kernels import pack_param


# Random parameter records.
def _params(rng, n):
    return b"".join(pack_param(rng.randrange(-5000, 5000), rng.randrange(1 << 30, 1 << 31), rng.randrange(-12, 0)) for _ in range(n))


# Several jobs through run_queue give the same outputs as the same jobs run one by one.
@pytest.mark.parametrize("lanes,rq", [(1, 0), (2, 1), (4, 1)])
def test_queue_matches_single_jobs(lanes, rq):
    rng = random.Random(5)
    # Two engines with identical contents: one runs a queue, the other single jobs.
    mk = lambda: HwEngine(SimBus(11, 9, 8, 10, lanes, rq, queue=True, perf=True))
    q, s = mk(), mk()
    jobs, singles = [], []
    xa = ya = wa = pa = 0
    for j in range(4):
        N, K, nvec = rng.randrange(1, 7), rng.randrange(1, 16), rng.randrange(1, 4)
        rows = (N + lanes - 1) // lanes
        x = bytes(rng.randrange(256) for _ in range(nvec * K))
        w = bytes(rng.randrange(256) for _ in range(N * K))
        p = _params(rng, N)
        for e in (q, s):
            e.upload_w(wa, w, K)
            e.upload_p(pa, p)
            e.write_x(xa, x)
        job = (K, N, nvec, K, wa, pa, 3, -2, -128, 127, 0, xa, ya)
        jobs.append(job)
        singles.append((job, x))
        xa, ya, wa, pa = xa + nvec * K, ya + nvec * N, wa + rows * K, pa + rows
    # Queue run: outputs stay in Y at each job's base.
    q.run_queue(jobs)
    # Single runs: each starts from the job's own bases.
    for job, x in singles:
        K, N, nvec = job[0], job[1], job[2]
        y_single = s.run(*job[:11], x, job[11], job[12])
        assert list(q.read_y(job[12], nvec * N)) == [v & 0xFF for v in y_single]
    # One start for the whole list is cheaper than one per job on the link.
    assert q.bus.frames < s.bus.frames


# Counters add up over a queue run and restart with the next start.
def test_perf_counters():
    rng = random.Random(1)
    e = HwEngine(SimBus(10, 9, 8, 8, 2, 1, queue=True, perf=True))
    K, N, nvec = 12, 5, 3
    e.upload_w(0, bytes(rng.randrange(256) for _ in range(N * K)), K)
    e.upload_p(0, _params(rng, N))
    e.write_x(0, bytes(rng.randrange(256) for _ in range(nvec * K)))
    job = (K, N, nvec, K, 0, 0, 0, 0, -128, 127, 0, 0, 0)
    e.run_queue([job, job])
    mac, stall, out, jobs = e.perf()
    # Groups per vector: ceil(5 / 2) = 3; every group issues K MAC clocks.
    assert mac == 2 * nvec * 3 * K
    assert out == 2 * nvec * N and jobs == 2
    # Utilisation is the MAC clocks over all clocks.
    assert 0 < mac / e.cycles < 1
    # The next start clears them.
    e.run(*job[:11], bytes(nvec * K), job[11], job[12])
    assert e.perf()[3] == 1


# Error paths: no queue in the build, too many jobs, a failing job reports how many finished.
def test_queue_errors():
    plain = HwEngine(SimBus(10, 9, 8, 8))
    with pytest.raises(RuntimeError):
        plain.run_queue([(1, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0, 0, 0)])
    e = HwEngine(SimBus(10, 9, 8, 8, queue=True))
    with pytest.raises(ValueError):
        e.run_queue([(1, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0, 0, 0)] * 9)
    good = (1, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0, 0, 0)
    bad = (0, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0, 0, 0)
    with pytest.raises(RuntimeError):
        e.run_queue([good, bad, good])
    assert e.bus.read(0x119) == 1

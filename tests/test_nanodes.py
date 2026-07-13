from collections import deque
from collections.abc import Iterator, Sequence
from random import Random
from time import sleep, time
from threading import get_ident
from unittest import TestCase

from nanodes import (
    BaseNode,
    Batcher,
    mapper,
    Loader,
    RoundRobin,
    Wrapper,
    seed_from,
    SerialMapper,
    ParallelMapper,
    SortedMerger,
)


class TestParallelMapperRuntime(TestCase):
    """
    More of a sanity check:

    ``sleepy_fn()`` sleeps ((0.1 + 0.3) / 2) = 0.2 seconds on average per sample. So the elapsed time should be:

    - ca. 0.2 * SAMPLES * EPOCHS with SerialMapper instead of ParallelMapper,
    - ca. 0.2 * SAMPLES * EPOCHS / WORKERS with ParallelMapper and `in_order=False`,
    - a number greater than but on the same order as the previous one with ParallelMapper and `in_order=True`.
    """

    BATCH_SIZE = 4
    EPOCHS = 7
    SAMPLES = 20
    WORKERS = 4

    @staticmethod
    def _loader(serial: bool, in_order: bool | None = None) -> Loader:

        rand = Random(42)

        class DummyNode(BaseNode):
            def __init__(self, num: int, name: str):
                super().__init__()
                self._num = num
                self._offset = 0
                self._name = name

            def iter(self) -> Iterator[str]:
                for i in range(self._num):
                    yield f"{self._name}.{i}"

        def sleepy_fn(val):
            sleep(s := rand.uniform(0.1, 0.3))
            return f"sleepy_fn (t {str(get_ident())[-4:]}: {val} (after {s:.2f} s)"

        num_workers = 0 if serial else TestParallelMapperRuntime.WORKERS
        node = DummyNode(TestParallelMapperRuntime.SAMPLES, "Dummy")
        node = mapper(node, fn=sleepy_fn, num_workers=num_workers, in_order=in_order)
        node = Batcher(node, batch_size=TestParallelMapperRuntime.BATCH_SIZE, drop_last=False)
        return Loader(node)

    def test_serial(self):
        loader = self._loader(serial=True)
        start = time()
        for epoch_ in range(self.EPOCHS):
            for _ in loader(regenerate=epoch_ != self.EPOCHS - 1):
                pass
        elapsed_is = time() - start
        elapsed_should = 0.2 * self.SAMPLES * self.EPOCHS
        self.assertTrue(
            abs(elapsed_is - elapsed_should) < 0.1 * elapsed_should, f"{elapsed_is=:.2f}, {elapsed_should=:.2f}"
        )

    def test_parallel_in_order(self):
        loader = self._loader(serial=False, in_order=True)
        start = time()
        for epoch_ in range(self.EPOCHS):
            for _ in loader(regenerate=epoch_ != self.EPOCHS - 1):
                pass
        elapsed_is = time() - start
        elapsed_should = 0.2 * self.SAMPLES * self.EPOCHS / self.WORKERS
        self.assertTrue(  # Still need a very high tolerance here, because of in-order processing
            abs(elapsed_is - elapsed_should) < 0.25 * elapsed_should, f"{elapsed_is=:.2f}, {elapsed_should=:.2f}"
        )

    def test_parallel_out_of_order(self):
        loader = self._loader(serial=False, in_order=False)
        start = time()
        for epoch_ in range(self.EPOCHS):
            for _ in loader(regenerate=epoch_ != self.EPOCHS - 1):
                pass
        elapsed_is = time() - start
        elapsed_should = 0.2 * self.SAMPLES * self.EPOCHS / self.WORKERS
        self.assertTrue(
            abs(elapsed_is - elapsed_should) < 0.1 * elapsed_should, f"{elapsed_is=:.2f}, {elapsed_should=:.2f}"
        )


class TestRandomReproducibility(TestCase):

    BATCH_SIZE = 3

    @staticmethod
    def _loader() -> Loader:

        class DummyNode(BaseNode):
            def __init__(self, num: int, name: str):
                super().__init__()
                self._num = num
                self._name = name
                self._factor = 1
                self._next_iter_seed = num

            def iter(self) -> Iterator[str]:
                iter_rand = Random(self._next_iter_seed)
                for i in range(self._num):
                    yield f"{self._name}.{i}.{self._factor * iter_rand.randrange(256):03d}"

            def regenerate(self):
                super().regenerate()
                self._next_iter_seed += 1

            def set_epoch(self, epoch):
                self._next_iter_seed += epoch

        class Hook:

            def __init__(self, seed: int):
                self._seed = seed
                self._instance_rng = Random(seed)
                self._next_call_seed = seed_from(self._instance_rng)

            def __call__(self, nodes: Sequence[DummyNode]):
                # For all nodes alter their seeds
                call_rng = Random(self._next_call_seed)
                for node in nodes:
                    node._next_iter_seed = seed_from(call_rng)
                self._next_call_seed = seed_from(self._instance_rng)

            def set_epoch(self, epoch):
                for _ in range(epoch):
                    self._next_call_seed = seed_from(self._instance_rng)

        nodes = [DummyNode(42, "Dummy42"), DummyNode(23, "Dummy23"), DummyNode(10, "Dummy10")]
        node = RoundRobin(
            nodes,
            shuffle=True,
            seed=0xC0FFEE,
            pre_epoch_hook=Hook(seed=123),
        )
        node = Batcher(node, batch_size=TestRandomReproducibility.BATCH_SIZE, drop_last=False)
        return Loader(node)

    def test_random_reproducibility(self):
        loader_1 = self._loader()
        loader_2 = self._loader()
        num_epochs = 5
        start_epoch_1 = 0  # Iterate over all
        start_epoch_2 = num_epochs - 1  # Last epoch only
        loader_1.set_epoch(start_epoch_1)
        for epoch_ in range(start_epoch_1, num_epochs):
            batches_1 = list(loader_1(regenerate=epoch_ != num_epochs - 1))
        loader_2.set_epoch(start_epoch_2)
        for epoch_ in range(start_epoch_2, num_epochs):
            batches_2 = list(loader_2(regenerate=epoch_ != num_epochs - 1))
        self.assertListEqual(batches_1, batches_2, f"Epoch {epoch_}: {batches_1} != {batches_2}")


class TestWrapper(TestCase):

    def _loader(self, name: str, force_exhaust_wrapped: bool):

        class ReproducibleIterable:
            def __init__(self, num_samples: int = 10):
                self._samples = deque(range(10))

            def __iter__(self):
                yield from self._samples
                self._samples.rotate(-1)

        return Loader(Wrapper(ReproducibleIterable(), force_exhaustion=force_exhaust_wrapped))

    def test_wrapper(self):

        loader_f = self._loader(force_exhaust_wrapped=False, name="f")
        loader_t1 = self._loader(force_exhaust_wrapped=True, name="t1")
        loader_t2 = self._loader(force_exhaust_wrapped=True, name="t2")
        num_epochs = 5
        start_epoch_t1 = 0
        start_epoch_t2 = num_epochs - 1
        loader_t1.set_epoch(start_epoch_t1)
        samples_t1_epoch_0 = None
        for epoch_ in range(start_epoch_t1, num_epochs):
            samples_t1 = list(loader_t1(regenerate=epoch_ != num_epochs - 1))
            if epoch_ == 0:
                samples_t1_epoch_0 = samples_t1
        loader_t2.set_epoch(start_epoch_t2)
        for epoch_ in range(start_epoch_t2, num_epochs):
            samples_t2 = list(loader_t2(regenerate=epoch_ != num_epochs - 1))
        self.assertListEqual(samples_t1, samples_t2, f"Epoch {epoch_}: {samples_t1} != {samples_t2}")

        # No effect of set_epoch here
        loader_f.set_epoch(start_epoch_t2)
        for epoch_ in range(start_epoch_t2, num_epochs):
            samples_f = list(loader_f(regenerate=epoch_ != num_epochs - 1))
        self.assertListEqual(samples_t1_epoch_0, samples_f, f"Epoch {epoch_}: {samples_t1_epoch_0} != {samples_f}")


class TestMapperOneToN(TestCase):

    def test(self):

        n = 3
        # SerialMapper
        for type_, num, io in zip([SerialMapper, ParallelMapper], [0, 1], [None, False]):
            m = mapper(range(n), fn=lambda i: [i, i**2], num_workers=num, one_to_n=2, in_order=io)
            self.assertIsInstance(m, type_)
            self.assertEqual(len(m), 2 * n)
            self.assertEqual(list(m), [x for pair in zip(range(n), (i**2 for i in range(n))) for x in pair])


class TestSortedMerger(TestCase):

    def test(self):

        m = SortedMerger(["abe", "ACDEF"], key=str.lower)  # Use `lower` to have tiebreaker take effect
        self.assertEqual("".join(m), "aAbCDeEF")


class TestLength(TestCase):

    def test_success(self):

        sources = [range(5), range(3)]  # -> total length: 8

        # Simple
        self.assertEqual(len(RoundRobin(sources)), 8)

        # Accumulated, batched, divisible
        l = Loader(Batcher(RoundRobin(sources, shuffle=False), batch_size=2, drop_last=True))
        self.assertEqual(len(l), 4)
        l = Loader(Batcher(RoundRobin(sources, shuffle=False), batch_size=2, drop_last=False))
        self.assertEqual(len(l), 4)

        # Accumulated, batched, *not* divisible
        l = Loader(Batcher(RoundRobin(sources, shuffle=False), batch_size=3, drop_last=True))
        self.assertEqual(len(l), 2)
        l = Loader(Batcher(RoundRobin(sources, shuffle=False), batch_size=3, drop_last=False))
        self.assertEqual(len(l), 3)

    def test_failure(self):

        with self.assertRaises(TypeError) as ctx:
            len(Loader(iter(range(5))))

        self.assertIn("length of wrapped", str(ctx.exception))

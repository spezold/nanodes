import atexit
from collections import deque
from collections.abc import Iterator, Sequence, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from itertools import islice
from logging import getLogger, NullHandler
from random import Random
from threading import RLock
from typing import Callable, final, overload, Self

__version__ = "0.1.0"

logger = getLogger(__name__)
logger.addHandler(NullHandler())


# Some helper functions


def _raise_if(cond: bool, exception: type[BaseException] = ValueError, *, msg: str):
    if cond:
        raise exception(msg)


def _pop_next_completed_from(futures: set[Future]) -> Future:
    el = next(as_completed(futures))
    futures.remove(el)
    return el


@overload
def seed_from(generator: Random) -> int: ...
@overload
def seed_from(generator: None) -> None: ...
def seed_from(generator: Random | None) -> int | None:
    """
    Draw an integer from the given random number generator that, in turn, can be used as a seed for another random
    number generator; do nothing of no generator is given (convenience function).
    :param generator: optional random number generator to draw from
    :return: resulting integer in ``[0,2**64)``; None, if no generator is given
    """
    return None if generator is None else generator.randrange(2**64)


# Actual node classes


class BaseNode[T]:

    def __init__(self, source: "BaseNode | None" = None):
        self._source = source
        self._exhausted = False

    def iter(self) -> Iterator[T]:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError(type[self])

    @final  # Increase protection from reimplementation (`iter()` should be implemented instead)
    def __iter__(self):
        _raise_if(self._exhausted, msg=f"{self.__class__.__name__} is exhausted. Is a `regenerate()` call missing?")
        yield from self.iter()
        self._exhausted = True

    def regenerate(self):
        """
        If overridden, must be called by all implementing nodes via ``super().regenerate()``
        """
        logger.debug(f"Regenerate {self.__class__.__name__}")
        if self._source is not None:
            self._source.regenerate()
        self._exhausted = False

    def set_epoch(self, epoch: int):
        """
        If overridden, must be called by all implementing nodes via ``super().set_epoch(epoch)``.

        Stateful nodes (e.g. those using random states) must ensure that the ``set_epoch(epoch)`` produces the same
        internal state at the start of the corresponding epoch, as would have been produced by reaching the epoch
        through standard iterations. The main reason is to enable reproducible behavior when resuming interrupted
        processing.

        :param epoch: zero-based epoch from which to start/continue processing
        """
        _raise_if(epoch < 0, msg=f"Need non-negative epoch number (got {epoch=})")
        logger.debug(f"Set epoch for {self.__class__.__name__}")
        if self._source is not None:
            self._source.set_epoch(epoch)


class Batcher(BaseNode):

    def __init__(self, source: BaseNode, *, batch_size: int, drop_last: bool):
        super().__init__(source)
        self._batch_size = batch_size
        self._drop_last = drop_last

    def iter(self) -> Iterator[list]:
        batch = []
        for item in self._source:
            batch.append(item)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if len(batch) and not self._drop_last:
            yield batch


class SerialMapper[S, T](BaseNode):

    def __init__(self, source: BaseNode[S], *, fn: Callable[[S], T]):
        super().__init__(source)
        self._fn = fn

    def iter(self) -> Iterator[T]:
        for item in self._source:
            yield self._fn(item)


class ParallelMapper[S, T](BaseNode):

    def __init__(self, source: BaseNode[S], *, fn: Callable[[S], T], num_workers: int, in_order: bool):
        super().__init__(source)
        _raise_if(num_workers < 1, msg=f"Need at least one worker (got num_workers={num_workers})")
        self._fn = fn
        self._num_workers = num_workers
        self._in_order = in_order

        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._lock = RLock()
        atexit.register(self._executor.shutdown)  # TODO: Really necessary? mapper+executor should live throughout run.

    def _locked_source(self) -> Iterator[S]:
        iterator = iter(self._source)
        exhausted = False
        while not exhausted:
            try:
                with self._lock:
                    item = next(iterator)
                yield item
            except StopIteration:
                exhausted = True

    def iter[C: "deque[Future[T]] | set[Future[T]]"](self) -> Iterator[T]:
        locked_source = self._locked_source()
        future_from: Callable[[S], Future[T]] = lambda el: self._executor.submit(self._fn, el)
        if self._in_order:
            cont_cls: type[C] = deque
            pop_next_from: Callable[[C], Future] = lambda cont: cont.popleft()
            append_to: Callable[[C, Future[T]], None] = lambda cont, el: cont.append(el)
        else:
            cont_cls: type[C] = set
            pop_next_from: Callable[[C], Future] = lambda cont: _pop_next_completed_from(cont)
            append_to: Callable[[C, Future[T]], None] = lambda cont, el: cont.add(el)
        # Pre-fill with up to `num_workers` tasks
        futures = cont_cls(future_from(el) for el in islice(locked_source, self._num_workers))
        while futures:  # Yield next result (from oldest task if in-order, from next completed task otherwise), refill
            yield pop_next_from(futures).result()  # `result()` will block without timeout if necessary
            try:
                append_to(futures, future_from(next(locked_source)))
            except StopIteration:
                pass  # Source exhausted → nothing more to submit


def mapper[S, T](
    source: BaseNode[S], *, fn: Callable[[S], T], num_workers: int, in_order: bool | None = None
) -> SerialMapper[S, T] | ParallelMapper[S, T]:
    _raise_if(num_workers < 0, msg=f"Need 0 or more workers (got num_workers={num_workers})")
    name = source.__class__.__name__
    if num_workers == 0:
        logger.debug(f"Create SerialMapper for {name}" + ("" if in_order is None else f" (ignore {in_order=})"))
        m = SerialMapper(source, fn=fn)
    else:
        _raise_if(in_order is None, msg=f"Got {in_order=} for {num_workers=}: need boolean value if num_workers > 0")
        logger.debug(f"Create ParallelMapper for {name}")
        m = ParallelMapper(source, fn=fn, num_workers=num_workers, in_order=in_order)
    return m


class RoundRobin[T](BaseNode):

    def __init__(
        self,
        sources: Sequence[BaseNode[T]],
        *,
        pre_epoch_hook: Callable[[tuple[BaseNode[T], ...]], None] | None = None,
        shuffle: bool = False,
        seed: int | Random | None = None,
    ):
        """
        In each full traversal of the node, cycle all source nodes until all are exhausted, as in the ``roundrobin``
        recipe of https://docs.python.org/3/library/itertools.html#itertools-recipes (20260521). With ``shuffle=True``,
        cycle all nodes in random order, drawing nodes independently for each new traversal and round. With
        ``shuffle=False``, cycle all nodes in the given order, as in the ``roundrobin`` recipe.

        Optionally add a pre-epoch hook that receives all source nodes in their given (unshuffled) order, offering the
        possibility to alter them before the start of an epoch.

        CAUTION: The hook is assumed to either (1) be stateless or (2) provide its own ``set_epoch(epoch)`` method to
        reproduce its state. Furthermore, (3) alterations of the hook are supposed to not accumulate across epochs, to
        ensure reproducibility when skipping actual epochs through ``set_epoch(epoch)``.

        :param sources: nodes to traverse
        :param pre_epoch_hook: pre-epoch hook (optional; default: None)
        :param shuffle: shuffle traversal order (True) or maintain provided order (False; default)
        :param seed: random seed (optional; default: None)
        """
        super().__init__(source=None)  # We (need to) keep track of the sources ourselves
        self._sources = tuple(sources)

        self._pre_epoch_hook = pre_epoch_hook
        self._instance_rng = (seed if isinstance(seed, Random) else Random(seed)) if shuffle else None
        self._next_iter_seed = seed_from(self._instance_rng)

    def iter(self) -> Iterator[T]:

        if self._pre_epoch_hook is not None:
            self._pre_epoch_hook(self._sources)

        iter_rng = None if self._next_iter_seed is None else Random(self._next_iter_seed)
        # Following ``roundrobin`` at https://docs.python.org/3/library/itertools.html#itertools-recipes (20260521)
        iterators = deque(iter(source) for source in self._sources)

        if iter_rng is None:
            rotate_iterators = lambda: iterators.rotate(-1)
        else:
            rotate_iterators = lambda: iterators.rotate(-int(iter_rng.randrange(max(1, len(iterators)))))
            rotate_iterators()  # Randomize for the first round

        while iterators:
            # Bring iterator of interest to front (rotate left), then pop it if exhausted or yield once from it if not
            try:
                yield next(iterators[0])
                rotate_iterators()  # Bring iterator of interest to front (rotate left) for next round
            except StopIteration:
                iterators.popleft()  # Pop exhausted
                if iter_rng is not None:
                    rotate_iterators()  # `None` (ordered): pop() brought next to front; `not None`: enforce shuffle

    def regenerate(self):
        # To be done by `super()`: (1) logging, (2) set `_exhausted` to False
        super().regenerate()
        # To be done by us: actually regenerating the sources and our own state
        for source in self._sources:
            source.regenerate()
        self._next_iter_seed = seed_from(self._instance_rng)

    def set_epoch(self, epoch: int):
        # To be done by `super()`: logging
        super().set_epoch(epoch)
        # To be done by us: set the epoch on the sources, on the hook if necessary, and call the seed function an
        # appropriate number of times to set our own state
        for source in self._sources:
            source.set_epoch(epoch)
        if self._pre_epoch_hook is not None and hasattr(self._pre_epoch_hook, "set_epoch"):
            self._pre_epoch_hook.set_epoch(epoch)
        for _ in range(epoch):
            self._next_iter_seed = seed_from(self._instance_rng)


class Prefetcher[T](BaseNode):
    """
    Fetch items from the source in a background thread, buffering up to ``prefetch_factor`` items ahead.
    """

    def __init__(self, source: BaseNode[T], *, prefetch_factor: int):
        super().__init__(source)
        _raise_if(prefetch_factor < 1, msg=f"Need at least a prefetch factor of 1 (got {prefetch_factor=})")
        self._prefetch_factor = prefetch_factor
        self._executor = ThreadPoolExecutor(max_workers=1)
        atexit.register(self._executor.shutdown)

    def iter(self) -> Iterator[T]:
        executor = self._executor
        iterator = iter(self._source)
        exhausted = False
        sentinel = object()

        def fetch_next():
            return next(iterator, sentinel)

        queue = deque(executor.submit(fetch_next) for _ in range(self._prefetch_factor))  # Pre-fill the buffer

        while not exhausted:
            item = queue.popleft().result()
            if item is sentinel:
                exhausted = True
            else:
                queue.append(self._executor.submit(fetch_next))  # Refill
                yield item


class Wrapper[T](BaseNode):

    def __init__(self, wrapped: Iterable[T], *, force_exhaustion: bool = False):
        """
        A wrapper class for any iterable (subject to the assumptions below).

        CAUTION: The iterable must conform to the following assumptions: (1) If ``force_exhaustion=False``, the
        iterable must, on each pass, provide the same sequence of samples. In this case, ``Wrapper.set_epoch(…)``
        becomes a no-op. (2) If ``force_exhaustion=True``, the iterable may, on each pass, provide a different sequence
        of samples; however, these sequences should still be deterministic in the sense that, each time the n-th pass
        is reached the n-th sequence should be the same. In this case, ``Wrapper.set_epoch(epoch=n)`` will exhaust the
        wrapped iterable ``n`` times before the next pass (which, potentially, might take a while).

        """
        super().__init__()
        self._wrapped = wrapped
        self._force_exhaustion = force_exhaustion

    def iter(self) -> Iterator[T]:
        yield from self._wrapped

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        if self._force_exhaustion:
            for _ in range(epoch):
                for _ in self._wrapped:
                    pass


class Loader[T]:

    def __init__(self, source: BaseNode[T]):
        self._source = source
        self._do_regenerate = True  # Optionally suppress regeneration

    def __iter__(self) -> Iterator[T]:
        yield from self._source
        if self._do_regenerate:
            self._source.regenerate()

    def __call__(self, *, regenerate: bool = True) -> Self:
        self._do_regenerate = regenerate
        return self

    def set_epoch(self, epoch: int):
        """
        Set the epoch for continued training.

        CAUTION: For ``epoch>0``, this assumes that the loader has not been called with ``regenerate=False`` and will
        fail otherwise.

        :param epoch: epoch number to start from (zero-based)
        """
        _raise_if(epoch < 0, msg=f"Need non-negative epoch number (got {epoch=})")
        _raise_if(epoch > 0 and not self._do_regenerate, msg=f"Can't set epoch: loader called with `regenerate=False`")
        self._source.set_epoch(epoch)

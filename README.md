# `datanodes` – A library for data processing pipelines

A lightweight, dependency-free alternative to
[`torchdata.nodes`](https://meta-pytorch.org/data/main/torchdata.nodes.html),
following the same compositing idea. Loading states is supported by means of `set_epoch()`, which enables reproducing
the internal state of a data pipeline at the beginning of the given epoch.

## Provided nodes

`BaseNode` is the abstract base class for all nodes, which can be subclassed to implement further custom nodes.
The following implementations are provided:

- `Batcher`: Batches the input node's data into lists of a given size.
- `SerialMapper` and `ParallelMapper`: Applies a given function to the input node's data, either serially or in
  parallel. The `mapper()` factory can be used to create a `SerialMapper` or `ParallelMapper` instance based on the
  given `num_workers` parameter.
- `RoundRobin`: Combines the data from multiple input nodes in a round-robin fashion.
- `Prefetcher`: Prefetches data from the input node in a separate thread, using a queue of a given size.
- `Wrapper`: A wrapper node that can be used to wrap any iterable data source, such as a list or a generator function.

The `Loader` class is technically not a node, but a node wrapper that provides automatic resets after exhaustion.

## Usage example

```python
from datanodes import Batcher, RoundRobin, Loader, Wrapper

wrappers = [Wrapper(range(5)), Wrapper("abc"), Wrapper("ABCDEFG")]
node = RoundRobin(wrappers, shuffle=True, seed=0xC0FFEE)
node = Batcher(node, batch_size=4, drop_last=False)
loader = Loader(node)

for epoch in range(3):
    collected_batches = []
    for batch in loader:
        collected_batches.append(batch)
    print(f"Epoch {epoch}: {collected_batches}")
```

Output:

```
Epoch 0: [['a', 'b', 0, 'A'], ['c', 1, 2, 3], [4, 'B', 'C', 'D'], ['E', 'F', 'G']]
Epoch 1: [[0, 1, 'a', 'A'], ['b', 'B', 'c', 2], [3, 'C', 'D', 'E'], ['F', 4, 'G']]
Epoch 2: [['a', 0, 1, 'b'], [2, 3, 4, 'c'], ['A', 'B', 'C', 'D'], ['E', 'F', 'G']]
```

Using `set_epoch()` on the `Loader` lets us continue where we left off:

```python
from datanodes import Batcher, RoundRobin, Loader, Wrapper

wrappers = [Wrapper(range(5)), Wrapper("abc"), Wrapper("ABCDEFG")]
node = RoundRobin(wrappers, shuffle=True, seed=0xC0FFEE)
node = Batcher(node, batch_size=4, drop_last=False)
loader = Loader(node)

START_FROM_EPOCH = 2

loader.set_epoch(START_FROM_EPOCH)

for epoch in range(START_FROM_EPOCH, 5):
    collected_batches = []
    for batch in loader:
        collected_batches.append(batch)
    print(f"Epoch {epoch}: {collected_batches}")
```

Output:

```
Epoch 2: [['a', 0, 1, 'b'], [2, 3, 4, 'c'], ['A', 'B', 'C', 'D'], ['E', 'F', 'G']]
Epoch 3: [[0, 1, 'A', 2], ['B', 'a', 3, 'C'], ['b', 'c', 4, 'D'], ['E', 'F', 'G']]
Epoch 4: [['A', 'a', 'b', 'B'], [0, 1, 'c', 'C'], ['D', 'E', 2, 'F'], ['G', 3, 4]]
```

# `nanodes` – A tiny library for data processing pipelines

The `nanodes`¹ ² ³ library is intended to be a lightweight, dependency-free alternative to
[`torchdata.nodes`](https://meta-pytorch.org/data/main/torchdata.nodes.html),
following the same compositing idea.

Loading states is supported by means of `set_epoch()`, which enables reproducing
the internal state of a data pipeline at the beginning of the given epoch.

<sub>¹ *nano* ∪ *nodes* = *nanodes*.  
² Also see *nanodes* in [*J. A. Jobling (ed.): The Key to Scientific Names*](https://birdsoftheworld.org/bow/key-to-scientific-names/search?q=nanodes).  
³ No, I do not speak [Japanese](https://en.wiktionary.org/wiki/なのです).⁴  
⁴ Yes, I like footnotes indeed.</sub>

## Provided nodes

`BaseNode` is the abstract base class for all nodes, which can be subclassed to implement further custom nodes.
The following implementations are provided:

- `Batcher`: Batches the input node's data into lists of a given size.
- `SerialMapper` and `ParallelMapper`: Applies a given function to the input node's data, either serially or in
  parallel, using multithreading in the latter case. The `mapper()` factory can be used to create a `SerialMapper` or
  `ParallelMapper` instance based on the given `num_workers` parameter.
- `RoundRobin`: Combines the data from multiple input nodes in a round-robin fashion, with or without shuffling the
  input nodes for each element.
- `Prefetcher`: Prefetches data from the input node in a separate thread, using a queue of given size.
- `Wrapper`: A wrapper node that can be used to wrap any iterable data source, as long as it is finite and can be
  iterated multiple times. Other than subclassing `BaseNode`, this is the easiest way to create a source node for a
  data pipeline. Even easier, iterable data sources are automatically wrapped for convenience when initializing any
  node or a loader. Wrapper nodes can likewise be created explicitly however, e.g. to control exhaustion behavior.

The `Loader` class is technically not a node, but a node wrapper that provides automatic resets after exhaustion.

## Usage example

```python
from nanodes import Batcher, RoundRobin, Loader, SerialMapper

sources = [range(5), "abc", "ABCDEFG"]
node = RoundRobin(sources, shuffle=True, seed=0xC0FFEE)               # shuffled round-robin
node = Batcher(node, batch_size=4, drop_last=False)                   # … to lists of size 4
node = SerialMapper(node, fn=lambda x: "".join(str(el) for el in x))  # … to strings
loader = Loader(node)

for epoch in range(3):
    collected_batches = []
    for batch in loader:
        collected_batches.append(batch)
    print(f"Epoch {epoch}: {collected_batches}")
```

Output:

```
Epoch 0: ['ab0A', 'c123', '4BCD', 'EFG']
Epoch 1: ['01aA', 'bBc2', '3CDE', 'F4G']
Epoch 2: ['a01b', '234c', 'ABCD', 'EFG']
```

Using `set_epoch()` on the `Loader` lets us continue where we left off:

```python
from nanodes import Batcher, RoundRobin, Loader, SerialMapper

sources = [range(5), "abc", "ABCDEFG"]
node = RoundRobin(sources, shuffle=True, seed=0xC0FFEE)               # shuffled round-robin
node = Batcher(node, batch_size=4, drop_last=False)                   # … to lists of size 4
node = SerialMapper(node, fn=lambda x: "".join(str(el) for el in x))  # … to strings
loader = Loader(node)

loader.set_epoch(2)

for epoch in range(2, 5):
    collected_batches = []
    for batch in loader:
        collected_batches.append(batch)
    print(f"Epoch {epoch}: {collected_batches}")
```

Output:

```
Epoch 2: ['a01b', '234c', 'ABCD', 'EFG']
Epoch 3: ['01A2', 'Ba3C', 'bc4D', 'EFG']
Epoch 4: ['AabB', '01cC', 'DE2F', 'G34']
```

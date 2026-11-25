[![CI](https://github.com/MichaelLT38/torch7/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelLT38/torch7/actions/workflows/ci.yml)

## Development Status

This is a maintained fork of [torch/torch7](https://github.com/torch/torch7), which
stopped active development around 2017 (its C backend lives on in PyTorch's ATen).
The goal here is a modern, fast CPU tensor library that stays **independent, small
and self-contained C**: no C++, no required dependencies, one supported compiler.

Out of scope: GPU, distributed training, a JIT, sparse tensors.

## Building

Requirements:

* **GCC >= 13** (the only supported compiler; on Windows use [MSYS2](https://www.msys2.org/) **UCRT64**)
* CMake >= 3.22
* LuaJIT (preferred) or Lua 5.1
* Optional: a BLAS/LAPACK such as OpenBLAS (otherwise TH uses built-in fallbacks
  and the linear-algebra functions that need LAPACK are unavailable)

`cwrap` and `paths` are bundled in `lib/`, so nothing else needs to be installed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLUA=$(command -v luajit)
cmake --build build
ctest --test-dir build --output-on-failure
```

`ctest` runs the TH C test suites (`th.<type>`, `th.misc`, `th.large`) and the Lua
suites (`lua.torch`, `lua.int64`, `lua.writeObject`, `lua.intops`). Without LAPACK, the 21 LAPACK-only Lua tests are
skipped and listed. `th.large` needs about 2 GB of RAM and skips itself if the
allocation fails.

### Compatibility with upstream torch7

* **64-bit sizes everywhere.** Sizes, strides, indices and `LongTensor`/`LongStorage`
  values are `int64_t` on every platform. Upstream used C `long`, which is only
  32 bits on Windows, so Windows builds were limited to 2^31 elements.
* **Binary files store longs as 8 bytes on every platform** (as Linux and macOS
  always did). Files written in binary mode by an upstream *Windows* build used
  4-byte longs; read them with `torch.load(path, 'b32')` or `file:longSize(4)`.
* **Loading is data-only by default.** `torch.load`, `torch.deserialize` and
  `File:readObject` refuse serialized Lua functions, because loading one runs code
  stored in the file (and LuaJIT does not verify bytecode). Tensors, storages,
  Torch classes, tables, numbers, strings and booleans load as before. For a file
  you trust, opt in with `torch.load(path, mode, referenced, {allowFunctions = true})`,
  `torch.deserialize(str, mode, {allowFunctions = true})` or
  `file:allowFunctions(true)`. Saving functions is unchanged.
* `LongTensor:random()` *with no arguments* on Windows now gives the same values as
  on Linux (upstream reduced them modulo 2^31 on Windows). Every other random
  stream is unchanged.
* **Integer arithmetic is fully defined** (upstream left these cases to C's
  undefined behavior, which on x86 masked shift counts or killed the process):
  * Shift amounts outside `[0, bits-1]` raise an error (a shift by `0` is the
    identity) for `lshift`, `rshift`, `clshift`, `crshift`, `arshift`, `carshift`.
  * Integer division by zero raises an error in `div`, `cdiv`, `fmod`, `cfmod`,
    `remainder`, `cremainder`, `floordiv`, `cfloordiv`; `MIN / -1` wraps to `MIN`.
  * Signed overflow wraps (two's complement): TH is compiled with `-fwrapv`.
* **New:** `arshift` / `carshift` (arithmetic right shift, `floor(x / 2^s)`) and
  `floordiv` / `cfloordiv` (floor division, pairing with `remainder`). `rshift`
  stays a *logical* shift on signed types, as upstream.

Installing with luarocks still works: `luarocks make rocks/torch-scm-1.rockspec`.

### Benchmarks

```sh
build/th_bench > new.csv
luajit bench/compare.lua bench/baselines/phase0_win11-ucrt64-gcc15_24t.csv new.csv
```

`bench/bench_pytorch.py` times the same operations in PyTorch for comparison.


## Need help? ##

Torch7 community support can be found at the following locations. As of 2019, the Torch-7 community is close to non-existent.

* Questions, Support, Install issues: [Google groups](https://groups.google.com/forum/#!forum/torch7)
* Reporting bugs: [torch7](https://github.com/torch/torch7/issues) [nn](https://github.com/torch/nn/issues) [cutorch](https://github.com/torch/cutorch/issues) [cunn](https://github.com/torch/cutorch/issues) [optim](https://github.com/torch/optim/issues) [threads](https://github.com/torch/threads/issues)
* Hanging out with other developers and users (strictly no install issues, no large blobs of text): [Gitter Chat](https://gitter.im/torch/torch7)

<a name="torch.reference.dok"></a>
# Torch Package Reference Manual #

__Torch__ is the main package in [Torch7](http://torch.ch) where data
structures for multi-dimensional tensors and mathematical operations
over these are defined. Additionally, it provides many utilities for
accessing files, serializing objects of arbitrary types and other
useful utilities.

<a name="torch.overview.dok"></a>
## Torch Packages ##

  * Tensor Library
    * [Tensor](doc/tensor.md) defines the _all powerful_ tensor object that provides multi-dimensional numerical arrays with type templating.
    * [Mathematical operations](doc/maths.md) that are defined for the tensor object types.
    * [Storage](doc/storage.md) defines a simple storage interface that controls the underlying storage for any tensor object.
  * File I/O Interface Library
    * [File](doc/file.md) is an abstract interface for common file operations.
    * [Disk File](doc/diskfile.md) defines operations on files stored on disk.
    * [Memory File](doc/memoryfile.md) defines operations on stored in RAM.
    * [Pipe File](doc/pipefile.md) defines operations for using piped commands.
    * [High-Level File operations](doc/serialization.md) defines higher-level serialization functions.
  * Useful Utilities
    * [Timer](doc/timer.md) provides functionality for _measuring time_.
    * [Tester](doc/tester.md) is a generic tester framework.
    * [CmdLine](doc/cmdline.md) is a command line argument parsing utility.
    * [Random](doc/random.md) defines a random number generator package with various distributions.
    * Finally useful [utility](doc/utility.md) functions are provided for easy handling of torch tensor types and class inheritance.

<a name="torch.links.dok"></a>
## Useful Links ##

  * [Community packages](https://github.com/torch/torch7/wiki/Cheatsheet)
  * [Torch Blog](http://torch.ch/blog/)
  * [Torch Slides](https://github.com/soumith/cvpr2015/blob/master/cvpr-torch.pdf)


# AlphaDev

This repository contains relevant pseudocode and algorithms for the publication
"Faster sorting algorithms discovered using deep reinforcement learning"

The repository contains two modules:

- `alphadev.py` with the pseudocode for the AlphaDev agent and the Assembly Game RL environment
- `sort_functions_test.cc` with the discovered assembly programs and checks their correctness. This includes the following:
    - `Sort3AlphaDev` sorts 3 elements with 17 instructions
    - `Sort4AlphaDev` sorts 4 elements with 28 instructions
    - `Sort5AlphaDev` sorts 5 elements with 43 instructions
    - `Sort6AlphaDev` sorts 6 elements with 57 instructions
    - `Sort7AlphaDev` sorts 7 elements with 76 instructions
    - `Sort8AlphaDev` sorts 8 elements with 91 instructions
    - `VarSort3AlphaDev` sorts up to 3 elements with 25 instructions
    - `VarSort4AlphaDev` sorts up to 4 elements with 57 instructions
    - `VarSort5AlphaDev` sorts up to 5 elements with 80 instructions

## Installation

The code present in `alphadev.py` is pseudocode to  simplify reproduction.
As such, no installation is required for the pseudocode.

To test the discovered assembly programs, we need to [install bazel](https://docs.bazel.build/versions/main/install.html) and verify it builds correctly (we only support Linux with clang, but other
platforms might work)

## Usage

The `alphadev.py` contains logic for the RL environment, AlphaDev agent and the
Assembly Game. The main components are:

- `AssemblyGame` This represents the Assembly Game RL environment. The state of
the RL environment contains the current program and the state of memory and
registers. Doing a step in this environment is equivalent to adding a new
assembly instruction to the program (see the `step` method). The reward is a
combination of correctness and latency reward after executing the assembly
program over an input distribution. For simplicity of the overall algorithm we
are not including the assembly runner, but assembly execution can be delegated
to an external library (e.g. [AsmJit](https://github.com/asmjit/asmjit)).
- `AlphaDevConfig` contains the main hyperparameters used for the AlphaDev
  agent. This includes configuration of AlphaZero, MCTS, and underlying
  networks.
- `play_game` contains the logic to run an AlphaDev game. This include the MCTS
procedure and the storage of the game.
- `RepresentationNet` and `PredictionNet` contain the implementation the
  networks used in the AlphaZero algorithm. It uses a [MultiQuery
  Transformer](https://arxiv.org/abs/1911.02150) to represent assembly
  instructions.


To run the assembly test in `sort_functions_test.cc`, use the following command:
`CC=clang bazel test :sort_functions_test`

## Citing this work

```bibtex
@Article{AlphaDev2023,
  author  = {Mankowitz, Daniel J. and Michi, Andrea and Zhernov, Anton and Gelmi, Marco and Selvi, Marco and Paduraru, Cosmin and Leurent, Edouard and Iqbal, Shariq and Lespiau, Jean-Baptiste and Ahern, Alex and Koppe, Thomas and Millikin, Kevin and Gaffney, Stephen and Elster, Sophie and Broshear, Jackson and Gamble, Chris and Milan, Kieran and Tung, Robert and Hwang, Minjae and Cemgil, Taylan and Barekatain, Mohammadamin and Li, Yujia and Mandhane, Amol and Hubert, Thomas and Schrittwieser, Julian and Hassabis, Demis and Kohli, Pushmeet and Riedmiller, Martin and Vinyals, Oriol and Silver, David},
  journal = {Nature},
  title   = {Faster sorting algorithms discovered using deep reinforcement learning},
  year    = {2023},
  volume  = {618},
  number  = {7964},
  pages   = {257--263},
  doi     = {10.1038/s41586-023-06004-9}
}
```


## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

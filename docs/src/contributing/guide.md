# [Contributing](@id contributing-guide)

Contributions to RxGPs.jl are welcome! This page covers how to get started.

## Setting up a development environment

1. Clone the repository:
   ```bash
   git clone https://github.com/ReactiveBayes/RxGPs.jl.git
   cd RxGPs.jl
   ```

2. Activate and instantiate the project:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. Run the test suite:
   ```julia
   Pkg.test()
   ```

## Project structure

| Directory | Contents |
|:----------|:---------|
| `src/` | Main source code |
| `src/SparseGaussianProcessnode/` | Node definitions and average energy computations |
| `src/rule/` | Message passing update rules, organised by node and edge |
| `src/helper_functions/` | Kernels, caching, approximation, prediction, meta, utilities |
| `test/node_rule/` | The rules and average energies, against each node's density (`reference.jl`) |
| `test/helper_functions/` | Kernels, approximations, cache, prediction and the M-step |
| `test/engine/` | Inference with RxInfer, end to end |
| `examples/` | Jupyter notebook examples |
| `docs/` | Documentation (Documenter.jl) |

## Adding a new message passing rule

1. Create or edit the appropriate file in `src/rule/<node_type>_rules/<edge>.jl`.
2. Define the rule with MessagePassingRulesBase's `@define_message_update_rule`, under the node's meta as its `algorithm`.
3. Add a case for it in `test/node_rule/`: a MessagePassingRulesTestUtils table whose expected value comes from the node's density in `test/node_rule/reference.jl`, or, for a rule that returns a log-density, a comparison with it at points.
4. Make sure `Pkg.test()` passes: after an unfiltered run it checks that every rule is selected by some test.

## Adding a new node

1. Define the node struct and declare it with `@define_factor_node` in `src/SparseGaussianProcessnode/`.
2. Define its average energy with `@define_average_energy`.
3. Implement rules for each edge.
4. Create a meta type for it if needed: a subtype of MessagePassingRulesBase's `AbstractAlgorithm`, immutable, with the rules declared under it as their `algorithm`. Working memory belongs in a rule's `scratch`, not in the meta.
5. Export the node and meta from the main module.
6. Add tests and update documentation.

## Building the documentation locally

From the repository root, `make docs` builds the site into `docs/build/`, and `make docs-serve`
serves it with live reload (in draft mode, which skips running the examples). Without `make`:

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()          # docs/Project.toml takes RxGPs from the repository (on Julia 1.10, also Pkg.develop(path = "."))
include("docs/make.jl")
```

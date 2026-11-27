# WSPax
### Weight-Space Programming in JAX

WSPax (pronounced wespax) is a functional JAX-native library for manipulating parameter spaces using manual, agentic, and learned weight-space transformations.

Built around [JAX](https://github.com/jax-ml/jax)'s PyTree abstraction and inspired by [Equinox](https://github.com/patrick-kidger/equinox)'s "models are just PyTrees" philosophy, WSPax is architecture-agnostic and independent of any particular neural network framework.

WSPax is currently in development and supports recording, applying, transferring, composing, inverting, and validating dense weight-space patches.

### References
- [Sparks of Weight-Space Programming](https://arxiv.org/abs/2611.09382)
- [Equinox: Neural Networks in JAX via Callable PyTrees and Filtered Transformations](https://arxiv.org/abs/2111.00254)
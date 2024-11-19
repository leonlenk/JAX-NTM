# JAX-NTM
A full remimplimentation of the paper [Neural Turing Machines](https://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne, and Ivo Danihelka in google's deep learning framework [JAX](https://jax.readthedocs.io/en/latest/quickstart.html).

## Terminology
- Algorithmic Tasks: has closed form solution
- Creative Tasks: solution is non-deterministic (i.e. image generation)
- Curiculum Training: iteratively updating the training distribution based on model performance

## Style Guide
### Naming Conventions
- Memory: external storage
- Controller: interfaces with memory
- Model: computation preforming

### Python
- Follow [PeP8](https://peps.python.org/pep-0008/)
- Add CI/CD for unit testing (Put test cases in bash script)
- Type Hints
- Use Global Config file

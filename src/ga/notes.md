# Genetic Algorithm ICM

Intrinsic curiosity module.
A single forward and backward ICM that allows exploration reward bonus for the network.

ICM:

2 submodule, a forward model and an inverse model.

A backbone model maps state s to ϕ(s) feature vector.
 - The forward model takes ϕ(s) and aₜ and tries to predict ϕ(sₜ₊₁).
 - The inverse model takes ϕ(s) and ϕ(sₜ₊₁) and tries to predict aₜ.

Optimization

 - The forward model is trained to minimize the distance between ϕ(sₜ₊₁)-pred and ϕ(sₜ₊₁)
 - The inverse model is trained to minimize the distance between aₜ-hat and aₜ.

The jointed optimization problem for the ICM module can be written as

 - min[ (1 - β) * Lᵢ + β * Lf ]

where Lᵢ is inverse model loss and Lf is forward model loss.

- Lᵢ = CrossEntropy(aₜ-pred, aₜ) (if discrete otherwise Norm2Sqr loss)
- Lf = Norm2Sqr(ϕ(sₜ₊₁)-pred - ϕ(sₜ₊₁))

The Intrinsic reward rₜ-i = γ * Norm2Sqr(ϕ(sₜ₊₁)-pred - ϕ(sₜ₊₁)).

------

# Consensus Crossover

- A, B parents
- Ma, Mb Memories of parents sampled during their last run

Randomly select A or B as start child, let M be the memories of the opposite parent
For k iterations train the child on M.

Variant with 2 childs generated where both A and B are selected as a starting point.

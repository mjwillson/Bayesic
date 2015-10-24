# Bayesic

Probabilistic programming for large datasets via a hybrid of variational inference algorithms.

Currently vapourware / early work-in-progress.

## Motivation

From a machine learning perspective, most probabilistic programming / Bayesian inference libraries I've come across suffer from one or more of these problems:

1. They don't scale to large or even medium-size datasets nor do they support online learning
2. They don't take advantage of the GPU
3. Inference is MCMC based, which even with recent advances like NUTS still seems a relatively slow and finicky way to fit a machine learning model when compared to optimisation-based approaches (and especially to stochastic optimisation)
4. Restrictions on model structure (no discrete latents, conjugate exponential family requirements, ...) limit their generality

Deep learning libraries on the other hand avoid problems 1,2,3 but typically fail completely at probabilistic modelling -- stochastic latent variables aren't supported at all, and for parameters only MAP or ML inference is supported.

Recent research on variational inference has answers, I think, but as far as I can see nobody's quite figured out how to piece it all together into a package that's great for real-world probabilstic machine learning,

## Sketch plan

* Use Python, with theano (or maybe CGT) for GPU support and symbolic differentiation.
* Allow arbitrary directed graphical models to be expressed (So far, so much like PyMC3), only restriction being that log-likelihoods must be differentiable with respect to any continuous variables. Similar approach to PyMC3 but allowing discrete latent variables.
* Exponential family distributions to implement an additional interface exposing the relevant details

### Exploit conjugate exponential family nodes where present

Identify nodes which have the conjugate exponential family property in the context of their markov blanket. Either:

* Collapse/marginalise these nodes (perhaps as an on-demand-only thing, or perhaps automatable), where a closed form is known for the compound distribution (seems to be the case for most(all?) exponential families under conjugate priors), or
* As an alternative to collapsing, consider an update optimising the KL-corrected bound as described in [5], or
* Approximate them with a variational distribution in the same family, and apply a traditional variational message passing update (equivalently a unit-step natural gradient update) for their variational parameters which is derivable automatically / in closed form. This is the approach taken by VIBES [1] and one of the inference options in its successor Infer.NET [2]
* Note this approach extends to the use of discrete latent variables to choose between alternative conjugate parents ([1] has a nice explanation)

### Discrete nodes with finite support

For any unobserved discrete variables that don't benefit from conjugacy, but have finite support, either:

* Marginalise them by summation (this might sometimes be an attractive option even when alternatives like collapsing the parameters are also available -- how to choose?), or
* Approximate by a general/unconstrained discrete variational distribution, and compute a mean-field update for the parameters using the general formula.
* Or would there ever be a reason to prefer a gradient-based update for this kind of parameter?

### Non-conjugate/non-exp-family continuous nodes

For any unobserved continuous variables that don't benefit from conjugacy or aren't exponential family, apply a stochastic update based on an estimate of the gradient -- either:

* Use the reparameterisation trick from the Black Box Variational Inference paper [4] and the Auto-encoding VB paper [8], or
* The variance-reduced estimator from [7] (although this requires training a neural net to generate a variance reduction term), or [9] (although I think this requires proposing a model-specific control variate / might be hard to automate)
* The Local Expectation Gradients approach from [6], which also works for discrete variables. This is shown to be lower-variance than [4] although requires more function evaluations for Gaussian quadrature, or
* Should probably pick just one of these -- whichever is most general without being too slow -- maybe [6] ?

### Discrete numeric nodes with infinite (or large) support

For any unobserved discrete variables with infinite support (Poisson, Negative binomial, Geometric...):

* Re-parameterisation trick won't work
* [7] or [9] could work but with the downsides mentioned
* [6] would require us to approximate the infinite sum by some discrete equivalent of Gaussian quadrature. Doable but details might be a bit fiddly, still potentially preferable
* These kind of variables don't tend to occur that much as latents anyway, so could omit support initially

### MAP inference / EM / delta-function approximations

Any node could alternatively be approximated by a delta function, leading to an EM-like M-step update for that node. In the conjugate case you should be able to do an exact M-step update to the conjugate posterior mode (?); in the non-conjugate case a gradient M-step update; in the discrete case an exact update via brute-force argmax (this would probably not be a good idea very often, although e.g. you might want a classifier's prediction as an argmax)

### Minibatch stochastic updates

Any nodes which have iid data as children could have stochastic gradient updates applied via subsampled minibatches. This would be doubly stochastic in the case of BBVI-style updates and would allow for online learning. As a special case, this together with MAP/delta-function approximations would also allow neural models to be fit in the usual way.
* This potentially applies after marginalisation too, e.g. in case of a mixture model where the latents were marginalised
* This would have to be something you explicitly ask for, specifying which axes on which tensors of iid observed variables correspond to the index of the datapoint.

Where there's an IID sub-model for each data point which includes some unmarginalised latents (e.g. a mixture model or LDA), a similar approach to the Stochastic Variational Inference paper [4] could be taken in order to achieve online learning / minibatch stochastic updates.
* For each batch you maximise the local variational parameters for the batch's latents, then you apply a stochastic update to the locally-maximised bound which updates the global parameters.
* In the paper they assume conjugate-exponential structure and use natural gradients, but the same approach should work for the regular (and already stochastic) gradients arising from BBVI or local expectation gradient style updates.
* Potentially some details to work out here.
* This would have to be something you explicitly ask for, specifying the variables in the submodel and the axes for the datapoint indices.
* Could also try the Variational Autoencoder approach from [8], although you'd probably have to help it in defining the structure of the inference network, might not be such a black-box approach.

## Deterministic nodes

Arbitrary differentiable deterministic functions of continuous variables should be allowable. They'll break conjugacy (or at least our ability to discover it automatically) so BBVI-style updates will be required. If deterministic functions of continuous variables are observed, we'd need the function to be invertible and its inverse function available, and its Jacobean too if you want the log-probability of the transformed data.

I think we could allow arbitrary deterministic functions of discrete latents too, thinking about them in terms of the pick/discrete choice functions described in [1] which can preserve conjugacy when they choose between conjugate parents (e.g. choosing between Dirichlet topic distributions for words in LDA). When discrete variables are marginalised or approximated by full discrete factors, I think we could manage deterministic functions of them too, although we might need the inverse of the deterministic implemented too, and it might be quite easy to write something that blows up.

Linear transformations of Gaussians are another kind of deterministic that it might be nice to have special handling for, in order to allow things like marginalisation and conjugacy to still work when you have hierarchical models with Gaussians in them. Again [1] mentions this.

## Papers

Sorry I'm crap at citing things properly in Markdown:

[1] VIBES: http://papers.nips.cc/paper/2172-vibes-a-variational-inference-engine-for-bayesian-networks.pdf

[2] Infer.NET: http://research.microsoft.com/en-us/um/cambridge/projects/infernet/

[3] Black box variational inference: http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf

[4] Stochastic Variational Inference: http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf

[5] Fast variational inference in the conjugate exponential family: http://papers.nips.cc/paper/4766-fast-variational-inference-in-the-conjugate-exponential-family.pdf

[6] Local Expectation Gradients for Doubly Stochastic Variational Inference http://arxiv.org/abs/1503.01494

[7] Neural Variational Inference and Learning in Belief Networks http://arxiv.org/pdf/1402.0030v2.pfd

[8] Auto-Encoding Variational Bayes http://arxiv.org/pdf/1312.6114v10.pdf

[9] Variational Bayesian Inference with Stochastic Search http://www.cs.columbia.edu/~blei/papers/PaisleyBleiJordan2012a.pdf
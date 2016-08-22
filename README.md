# SGCRFpy:
## Sparse Gaussian Conditional Random Fields in Python

SGCRFpy is a Python implementation of Sparse Gaussian Conditional Random Fields (CRF) with a familiar API. CRFs are discriminative graphical models that are useful for performing inference when output variables are known to obey a structure.

## Background
A Gaussian CRF models the conditional probability density of `y` given `x` as

![alt tag](http://latex.codecogs.com/svg.latex?p(y|x;\\Lambda,\\Theta) = \\frac{e^{-y^\\top \\Lambda y -2 x^\\top \\Theta y}}{Z(x)})

where

![equation](http://latex.codecogs.com/svg.latex?Z(x) = c |\\Lambda|^{-1} e^{x^\\top \\Theta \\Lambda ^{-1} \\Theta ^\\top x})

This is equivalent to:

![equation](http://latex.codecogs.com/svg.latex?y|x\\sim\\mathcal{N}(\\Theta \\Lambda^{-1}x,\\Lambda^{-1}))

which is a reparametrization of standard linear regression [1]

In this sense, `Lambda` models the structure between output variables `y`, while `Theta` models the direct relationships between `x` and `y`. For example, in genetical genomics, a gene network `Lambda` controls how genetic perturbations in `Theta` propagate indirectly to other gene-expression traits [1]. In wind-farm power forecasting, `Lambda` models the spatial and temporal correlations between various generators, while `Theta` captures the conditional dependencies on exogenous variables [2].

Sparse Gaussian CRFs are a particular flavor of Gaussian CRFs where the loss function includes an `L1` penalty in order to promote sparsity among the estimated parameters. Setting `lam L` >> `lam T` results in Lasso regression, while setting `lam T` >> `lam L` results in Graphical Lasso.

<img src=images/random_graph.png height=70% width=70%>

## API
The API is simple and familiar and leads to one-liners:

```python
from sgcrf import SparseGaussianCRF

model = SparseGaussianCRF()
model.fit(X_train, Y_train).predict(X_test)
```

Since the model is probabilistic, it's also easy to generate lots of samples of `y` given `x`:

```python
Y = model.sample(X, n=100000)
```

The API is inspired by [Keras](https://github.com/fchollet/keras) which allows continued model training, so you can inspect your model...

```python
model = SparseGaussianCRF(learning_rate=0.1, n_iter=5)
model.fit(X_train, Y_train)
loss = model.lnll
plt.plot(loss)
```
<img src=images/training_a.png height=60% width=60%>

...and pick up where you left off:

```python
model.set_params(learning_rate=1)
model.fit(X_train, Y_train)
loss += model.lnll
plt.plot(loss)
```
<img src=images/training_b.png height=60% width=60%>

## Optimization Details
Optimization is performed via alternating Newton coordinate descent of the regularized negative log-likelihood [4],  which significantly reduces the computation time compared to previous methods [1][2][3]. The optimization alternates between updating `Lambda` given `Theta` using a second-order approximation to the objective, and then updating `Theta` given `Lambda`, which requires no Taylor series approximation.

Notable features:  

- Newton steps are solved via **coordinate descent** because the problem includes an `L1` penalty.
- Parameter updates are restricted to an **active set** of variables which produces a substantial speedup for sparse problems.
- Frequently used large matrix products are **cached**, and only their rows and columns are updated after coordinate descent updates.
- The step size for `Lambda` is chosen via **line search**.


## References
0. Lingxue Zhang, Seyoung Kim 2014  
Learning Gene Networks under SNP Perturbations Using eQTL Datasets  
http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003420


0. Wytock and Kolter 2013  
Probabilistic Forecasting using Sparse Gaussian CRFs  
http://www.zicokolter.com/wp-content/uploads/2015/10/wytock-cdc13.pdf


0. Wytock and Kolter 2013  
Sparse Gaussian CRFs Algorithms Theory and Application  
https://www.cs.cmu.edu/~mwytock/papers/gcrf_full.pdf


0. McCarter and Kim 2015  
Large-Scale Optimization Algorithms for Sparse CGGMs  
http://www.jmlr.org/proceedings/papers/v51/mccarter16.pdf


0. McCarter and Kim 2016  
On Sparse Gaussian Chain Graph Models  
http://papers.nips.cc/paper/5320-on-sparse-gaussian-chain-graph-models.pdf


0. Klinger and Tomanek 2007  
Classical Probabilistic Models and Conditional Random Fields  
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.4527&rep=rep1&type=pdf


0. Tong Tong Wu and Kenneth Lange 2008  
Coordinate Descent Algorithms for Lasso Penalized Regression  
http://arxiv.org/pdf/0803.3876.pdf

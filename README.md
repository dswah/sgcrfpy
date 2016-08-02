# SGCRFpy:
## Sparse Gaussian Conditional Random Fields in Python

SGCRFpy is a Python implementation of Sparse Gaussian Conditional Random Fields (CRF) with a familiar API. CRFs are discriminative models that are useful for performing inference where output variables are known to obey a structure.

A Gaussian CRF models the conditional probability density of `y` given `x` as

![alt tag](http://latex.codecogs.com/svg.latex?p(y|x;\\Lambda,\\Theta) = \\frac{e^{-y^\\top \\Lambda y -2 x^\\top \\Theta y}}{Z(x)})

where

![equation](http://latex.codecogs.com/svg.latex?Z(x) = c |\\Lambda|^{-1} e^{x^\\top \\Theta \\Lambda ^{-1} \\Theta ^\\top x})

This is equivalent to:

![equation](http://latex.codecogs.com/svg.latex?p(y|x)\\sim\\mathcal{N}(\\Theta \\Lambda^{-1}x,\\Lambda^{-1}))

In this sense, `Lambda` models the structure between output variables `y`, while `Theta` models the relationship between `x` and `y`.

Sparse Gaussian CRFs are a particular flavor of Gaussian CRFs where the loss function includes an `L1` penalty in order to promote sparsity among the estimated parameters.

The API is easy and familiar and leads to one-liners:
```python
from sgcrf import SparseGaussianCRF

model = SparseGaussianCRF()
model.fit(X_train, y_train).predict(X_test)
```

<img src=https://github.com/dswah/sgcrfpy/blob/master/images/random_graph.png height=60% width=60%>

## References
Lingxue Zhang, Seyoung Kim 2014  
Learning Gene Networks under SNP Perturbations Using eQTL Datasets  
http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003420


Wytock and Kolter 2013  
Probabilistic Forecasting using Sparse Gaussian CRFs  
http://www.zicokolter.com/wp-content/uploads/2015/10/wytock-cdc13.pdf


Wytock and Kolter 2013  
Sparse Gaussian CRFs Algorithms Theory and Application  
https://www.cs.cmu.edu/~mwytock/papers/gcrf_full.pdf


McCarter and Kim 2015  
Large-Scale Optimization Algorithms for Sparse CGGMs  
http://arxiv.org/pdf/1509.04681.pdf


McCarter and Kim 2016  
On Sparse Gaussian Chain Graph Models  
http://papers.nips.cc/paper/5320-on-sparse-gaussian-chain-graph-models.pdf


Klinger and Tomanek 2007  
Classical Probabilistic Models and Conditional Random Fields  
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.4527&rep=rep1&type=pdf


Tong Tong Wu and Kenneth Lange 2008  
Coordinate Descent Algorithms for Lasso Penalized Regression  
http://arxiv.org/pdf/0803.3876.pdf

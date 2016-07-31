# SGCRFpy:
## Sparse Gaussian Conditional Random Fields in Python

SGCRFpy is a Python implementation of Sparse Gaussian Conditional Random Fields (CRF) with a familiar API. CRFs are useful for performing inference where output variables are known to obey a structure

A Gaussian CRF models conditional probability density of y given x as

![alt tag](http://latex.codecogs.com/svg.latex?p(y|x\\Lambda\\Theta) = \\frac{e^{-y^\\top \\Lambda y -2 x^\\top \\Theta y}}{Z(x)})

where

![equation](http://latex.codecogs.com/svg.latex?Z(x) = c |\\Lambda|^{-1} e^{x^\\top \\Theta \\Lambda ^{-1} \\Theta ^\\top x})


This is equivalent to:

![equation](http://latex.codecogs.com/svg.latex?p(y|x)\\sim\\mathcal{N}(\\Theta \\Lambda^{-1}x,\\Lambda^{-1}))


The API is easy and familiar and leads to one-liners:
```
from sgcrf import SparseGaussianCRF

model = SparseGaussianCRF()
model.fit(X_train, y_train).predict(X_test)
```

![alt tag](https://github.com/dswah/sgcrfpy/blob/master/images/scgrf_random_graph.png)

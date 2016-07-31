# SGCRFpy: Sparse Gaussian Conditional Random Fields in Python

A Gaussian CRF models conditional probability density of y given x as

<p align="center">
  ![equation](http://latex.codecogs.com/svg.latex?p(y|x\\Lambda\\Theta) = \\frac{e^{-y^\\top \\Lambda y -2 x^\\top \\Theta y}}{Z(x)})
</p>

where

<center>
![equation](http://latex.codecogs.com/svg.latex?Z(x) = c |\\Lambda|^{-1} e^{x^\\top \\Theta \\Lambda ^{-1} \\Theta ^\\top x})
</center>


This is equivalent to:

<center>
![equation](http://latex.codecogs.com/svg.latex?p(y|x)\\sim\\mathcal{N}(\\Theta \\Lambda^{-1}x,\\Lambda^{-1}))
</center>

<center>![alt tag](https://github.com/dswah/sgcrfpy/blob/master/images/scgrf_random_graph.png)
</center>

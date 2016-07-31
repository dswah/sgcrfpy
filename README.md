# SGCRFpy: Sparse Gaussian Conditional Random Fields in Python

latexImg = function(latex){

    link = paste0('http://latex.codecogs.com/gif.latex?',
           gsub('\\=','%3D',URLencode(latex)))

    link = gsub("(%..)","\\U\\1",link,perl=TRUE)
    return(paste0('![](',link,')'))
}

`r latexImg('a = \\frac{b}{c}')`

![alt tag](https://github.com/dswah/sgcrfpy/blob/master/images/scgrf_random_graph.png)

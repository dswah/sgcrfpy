#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy import random as rng

from sgcrf import SparseGaussianCRF

def test_theta_0():
    rng.seed(0)
    n_samples = 100
    Y = rng.randn(n_samples, 5)
    X = rng.randn(n_samples, 5)

    sgcrf = SparseGaussianCRF(lamL=0.01, lamT=0.01)
    sgcrf.fit(X, Y)

    assert np.allclose(sgcrf.Lam, np.eye(5), .1, .2)

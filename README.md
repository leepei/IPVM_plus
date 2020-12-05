# IPVM+ - An Improved Inexact Proximal-Newton-Type Method Utilizing Manifold Identification for Partly Smooth Regularized Optimization

This code implements the algorithm proposed in the following paper in C/C++.
_LEE Ching-pei. [On the Iterate Convergence and Manifold Identification of
Inexact Proximal-Newton-Type Methods Under a Sharpness Condition](http://www.optimization-online.org/DB_FILE/2020/12/8143.pdf). 2020._

## Getting started
To compile the code, you will need a C++ compiler and BLAS and LAPACK libraries.
To build, please specify suitable library options in Makefile and input the following in a Linux console.

```
$ make
```

Then the program `./train` solves the optimization problem to obtain a model, and the program `./predict` uses the model to predict on testing data.

## Program Structure
Our code uses the same interface as [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/).
We provide L1-norm-regularized problems as example, but the general framework can be applied to other regularized problem classes as well.
One just needs to provide related function implementation realizing the `regularized_fun` class in `linear.cpp`.
A large part of the code is extended from our previous projects [MADPQN](http://www.github.com/leepei/madpqn) and [DPLBFGS](http://www.github.com/leepei/dplbfgs).

## Problem being solved

The code solves the following problems that have a dimensionality of `n` and the inputs are of the LIBSVM format.
(1): the L1-regularized logistic regression problem.

```
min_{w} |w|_1 + C \sum_{i=1}^l \log(1 + \exp(- y_i w^T x_i))
```

with `y_i` being either `-1` or `+1` and with a user-specified parameter `C > 0`,

(2): the L1-Regularized least-sqaure regression (LASSO) problem.

```
min_{w} |w|_1 + C \sum_{i=1}^l (w^T x_i - y_i)^2 / 2.
```

with `y_i` being real numbers and with a user-specified parameter `C > 0`, and

(3): the L1-regularized L2-loss support vector classification problem.

```
min_{w} |w|_1 + C \sum_{i=1}^l \max\{0,1 - y_i w^T x_i, 0\}^2.
```

with `y_i` being either `-1` or `+1` and with a user-specified parameter `C > 0`.

## Parameters

The parameter `C` in the formulation listed above is specified through `-c` in `train`.

In the `IPVM+_LBFGS` solver, we use the previous update directions and gradient differences to construct the approximation of the Hessian of the smooth part. The parameter `m` decides that we use the latest `m` iterations to construct such an approximation.

The algorithm is terminated when `Q_t <= \epsilon Q_0` for a given epsilon, where `Q_t` is the objective of the subproblem at the t-th outer iteration. This parameter is specified by `-e` in `train`.

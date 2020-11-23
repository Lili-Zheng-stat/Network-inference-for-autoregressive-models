# Network inference for autoregressive modelsi
## Introduction
 This repository contains codes for conducting estimation and testing for network parameters in high-dimensional autoregressive models. 
## Hypothesis testing for high-dimensional linear AR(p) model
The folder **linear-testing** includes R functions for conducting hypothesis testing for autoregressive parameters in high-dimensional AR(p) model, together with a tutorial **Tutorial.md** for applying the R functions.

The testing method is proposed in the paper [Testing for high-dimensional network parameters in auto-regressive models](https://projecteuclid.org/euclid.ejs/1576119708) by Lili Zheng and Garvesh Raskutti.

## Estimating context-dependent network from autoregressive point processes 
The folder **context-dependent-network** includes Matlab functions for estmating context-dependent network from marked point processes, where past events associated with one node can triger/inhibit future events of its neighbors and the marks of events modulates the influences among nodes. A tutorial **Tutorial.md** with several examples is also included in this folder.

The algorthms are proposed in the paper [Context-dependent self-exciting point processes: models, methods, and risk bounds in high dimensions](https://arxiv.org/abs/2003.07429) by Lili Zheng, Garvesh Raskutti, Rebecca Willett and Ben Mark. Real data sets and R codes for network visualization are included in **context-dependent-network/datasets-and-processing-codes**.    


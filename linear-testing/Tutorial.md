Testing for autoregressive network parameter
================
Lili Zheng

## Introduction

This is a tutorial for how to use the R functions in **Test\_function.R** to conduct hypothesis testing for autoregressive parameters in high-dimenisonal linear AR(1) model, using the methods proposed in our paper [Testing for high-dimensional network parameters in auto-regressive models](https://projecteuclid.org/euclid.ejs/1576119708). The paper covers testing methods and theoretical guarantees for general AR(p) models, while for simplicity of presentation, we only consider AR(1) model here.

Specifically, we consider the following time series model:
*X*<sub>*t* + 1</sub> = *A*<sup>\*</sup>*X*<sub>*t*</sub> + *ϵ*<sub>*t*</sub>.
 Based on data ![(X\_t)\_{t=1-p}^{T}](http://chart.apis.google.com/chart?cht=tx&chl=%28X_t%29_%7Bt%3D1-p%7D%5E%7BT%7D "(X_t)_{t=1-p}^{T}"), we test the hypothesis of whether a subset of entries (![d](http://chart.apis.google.com/chart?cht=tx&chl=d "d") entries) in ![A^\*](http://chart.apis.google.com/chart?cht=tx&chl=A%5E%2A "A^*") equal to an arbitrary vector ![\\mu\\in \\mathbb{R}^d](http://chart.apis.google.com/chart?cht=tx&chl=%5Cmu%5Cin%20%5Cmathbb%7BR%7D%5Ed "\mu\in \mathbb{R}^d"). We propose two test statistics ![\\tilde{U}\_{T}](http://chart.apis.google.com/chart?cht=tx&chl=%5Ctilde%7BU%7D_%7BT%7D "\tilde{U}_{T}") and ![\\hat{R}\_{T}](http://chart.apis.google.com/chart?cht=tx&chl=%5Chat%7BR%7D_%7BT%7D "\hat{R}_{T}") that both converge in distribution to ![\\chi\_d^2](http://chart.apis.google.com/chart?cht=tx&chl=%5Cchi_d%5E2 "\chi_d^2") when the tested entries equal to ![\\mu](http://chart.apis.google.com/chart?cht=tx&chl=%5Cmu "\mu").

## Example

Consider the following example where the matrix ![A\\in \\mathbb{R}^{M\\times M}](http://chart.apis.google.com/chart?cht=tx&chl=A%5Cin%20%5Cmathbb%7BR%7D%5E%7BM%5Ctimes%20M%7D "A\in \mathbb{R}^{M\times M}") is a block matrix: <img src="A_mat.png" width="50%" />

and ![M=30](http://chart.apis.google.com/chart?cht=tx&chl=M%3D30 "M=30"), the time series data is of size ![N=500](http://chart.apis.google.com/chart?cht=tx&chl=N%3D500 "N=500").

``` r
source("Test_functions.R")
M=30;N=500;
A0=matrix(c(1/4,1/2,1/2,1/4),2,2)
A=matrix(rep(0,M^2),M,M)
for(i in 1:(M/nrow(A0)))
{
  A[c(nrow(A0)*i-1,nrow(A0)*i),c(nrow(A0)*i-1,nrow(A0)*i)]=A0
}
var=1;
X=data.generate(M,N,A,"unif",var)
```

Suppose we want to test the hypothesis that ![A\_{1,1:2}=(0.25,0.5)](http://chart.apis.google.com/chart?cht=tx&chl=A_%7B1%2C1%3A2%7D%3D%280.25%2C0.5%29 "A_{1,1:2}=(0.25,0.5)"), ![A\_{2,3:4}=(0,0)](http://chart.apis.google.com/chart?cht=tx&chl=A_%7B2%2C3%3A4%7D%3D%280%2C0%29 "A_{2,3:4}=(0,0)") (which is true), then the following codes would give us two test statistics ![\\tilde{U}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Ctilde%7BU%7D_N "\tilde{U}_N") and ![\\hat{R}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Chat%7BR%7D_N "\hat{R}_N").

``` r
test=list(c(1,1,2),c(2,3,4));
mu=list(c(0.25,0.5),c(0,0));
ts=test_statistic(M,N,X,test,mu)
```

Here, each element of `test` specifies the tested entries in one row. In this example, `c(1,1,2)` in `test` means the first and second entries in the first row, while `c(2,3,4)` means the third and fourth entries of the second row. `ts` is a 2-dimensional vector where the first entry is ![\\tilde{U}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Ctilde%7BU%7D_N "\tilde{U}_N") and the second entry is ![\\hat{R}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Chat%7BR%7D_N "\hat{R}_N").

![\\tilde{U}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Ctilde%7BU%7D_N "\tilde{U}_N") and the corresponding ![p](http://chart.apis.google.com/chart?cht=tx&chl=p "p")-value:

``` r
ts[1]
```

    ## [1] 6.970791

``` r
1-pchisq(ts[1],df=4)
```

    ## [1] 0.1374399

![\\hat{R}\_N](http://chart.apis.google.com/chart?cht=tx&chl=%5Chat%7BR%7D_N "\hat{R}_N") and the corresponding ![p](http://chart.apis.google.com/chart?cht=tx&chl=p "p")-value:

``` r
ts[2]
```

    ## [1] 6.970791

``` r
1-pchisq(ts[2],df=4)
```

    ## [1] 0.1374399

Both ![p](http://chart.apis.google.com/chart?cht=tx&chl=p "p")-values are not significant, so we are not confident to reject the hypothesis using either of them. If we want to test the hypothesis that ![A\_{1,3:5}=(0.25,0.25,0)](http://chart.apis.google.com/chart?cht=tx&chl=A_%7B1%2C3%3A5%7D%3D%280.25%2C0.25%2C0%29 "A_{1,3:5}=(0.25,0.25,0)"), ![A\_{3,3:4}=(0.25,0)](http://chart.apis.google.com/chart?cht=tx&chl=A_%7B3%2C3%3A4%7D%3D%280.25%2C0%29 "A_{3,3:4}=(0.25,0)") (which is false), then the test statistics can be calculated as

``` r
test=list(c(1,3,4,5),c(3,3,4));
mu=list(c(0.25,0.25,0),c(0.25,0));
ts=test_statistic(M,N,X,test,mu)
ts[1]
```

    ## [1] 471.2835

``` r
ts[2]
```

    ## [1] 470.4669

Since we are testing for 5 entries, we should compare the test statistics to ![\\chi\_5^2](http://chart.apis.google.com/chart?cht=tx&chl=%5Cchi_5%5E2 "\chi_5^2") to get the ![p](http://chart.apis.google.com/chart?cht=tx&chl=p "p")-value:

``` r
1-pchisq(ts[1],df=5)
```

    ## [1] 0

``` r
1-pchisq(ts[2],df=5)
```

    ## [1] 0

Both ![p](http://chart.apis.google.com/chart?cht=tx&chl=p "p")-values are significant, suggesting that the hypothesis should be rejected regardless of which test statistic we use.

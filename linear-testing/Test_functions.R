library(glmnet)
#Function for generating linear AR(p) time series data
data.generate=function(M,N,A,ErrType,var)
{
  #M is the dimension of X^t, N is the sample size, 
  #A is the autoregressive parameter of size M*M
  #ErrType specifies the noise distribution: "unif" (uniform) or "Gaussian".
  #var is the variance of each entry of the noise term
  if(ErrType=="unif")
  {
    #First obtain initial X following stationary distribution
    X=runif(M,min=-1,max=1)*sqrt(3*var)
    for(i in 1:10000)
    {
      X=A%*%X+runif(M,min=-1,max=1)*sqrt(3*var)
    }
    #Generate future data 
    for(t in 1:N)
    {
      X=cbind(X,A%*%X[,t]+runif(M,min=-1,max=1)*sqrt(3*var))
    }
  }
  else if(ErrType=="Gaussian")
  {
    #Compute the stationary covariance Sigma
    Theta=(diag(M)-A%*%A)/var
    Sigma=solve(Theta)
    L=svd(Sigma);sqrtS=(L$u)%*%diag(sqrt(L$d))%*%t(L$v)
    X=sqrtS%*%rnorm(M)
    for(t in 1:N)
    {
      X=cbind(X,A%*%X[,t]+rnorm(M,sd=sqrt(var)))
    }
  }
  else{
    print('Wrong ErrType!')}
  return(X)
}

#R functions required for conducting hypothesis testing
est.A=function(M,N,r,X,row,C_list)
{
  #Estimate certain rows of A specified by vector row with size r, using Lasso estimator
  lambda_list=C_list*sqrt(log(M)/N)/r #devide C_list by r since the dimension of Y is rN
  loss=rep(0,length(lambda_list))
  cv.n=N/10
  for(i in 1:cv.n)
  {
    #Use glmnet to fit time series \{X_t\}_{t=i}^{N-cv.n+i}
    Z=matrix(rep(0,M*(N-cv.n)*r^2),r*(N-cv.n),r*M)
    for(k in 1:r)
    {
      Z[((N-cv.n)*(k-1)+1):((N-cv.n)*k),(M*(k-1)+1):(M*k)]=t(X)[i:(N-cv.n-1+i),]
    }
    Y=as.vector(t(X)[(i+1):(N-cv.n+i),row])
    fitA=glmnet(Z,Y,lambda=lambda_list*sqrt(N/(N-cv.n)),intercept=F)
    #Calculate the predictive loss
    for(j in 1:length(lambda_list))
    {
      beta=rep(0,M*r);beta[coef(fitA,s=lambda_list[j]*sqrt(N/(N-cv.n)))@i]=coef(fitA,s=lambda_list[j]*sqrt(N/(N-cv.n)))@x
      Ah=t(matrix(beta,M,r))
      test.set=(1:N)[-(i:(N-cv.n-1+i))]
      loss[j]=loss[j]+sum((X[row,test.set+1]-Ah%*%X[,test.set])^2)
    }
  }
  lambda=lambda_list[which.min(loss)]
  #Fit the whole data set
  Z=matrix(rep(0,M*N*r^2),r*N,r*M)
  for(i in 1:r)
  {
    Z[(N*(i-1)+1):(N*i),(M*(i-1)+1):(M*i)]=t(X)[1:N,]
  }
  Y=as.vector(t(X)[2:(N+1),row])
  fitA=glmnet(Z,Y,lambda=lambda,intercept=F)
  beta=rep(0,M*r);beta[coef(fitA)@i]=coef(fitA)@x
  Ah=t(matrix(beta,M,r))
  return(Ah)
}


est.w=function(M,N,k,X,tset,C_list)
{
  #Estimate w: the regression parameter of variables in tset (of size k) upon the rest variables
  #Estimation is conducted using the Lasso estimator. C_list includes the cross validation list of tuning parameters
  lambda_list=C_list*sqrt(log(M)/N)/k
  loss=rep(0,length(lambda_list))
  cv.n=N/10
  for(i in 1:cv.n)
  {
    #Use glmnet to fit time series \{X_t\}_{t=i}^{N-cv.n-1+i}
    Z=matrix(rep(0,(N-cv.n)*(M-k)*k^2),k*(N-cv.n),k*(M-k))
    for(n in 1:k)
    {
      Z[((N-cv.n)*(n-1)+1):((N-cv.n)*n),((M-k)*(n-1)+1):((M-k)*n)]=t(X)[i:(N-cv.n-1+i),-tset]
    }
    Y=as.vector(t(X)[i:(N-cv.n-1+i),tset])
    fitw=glmnet(Z,Y,lambda=lambda_list*sqrt(N/(N-cv.n)),intercept=F)
    #Calculate the predictive loss
    for(j in 1:length(lambda_list))
    {
      beta=rep(0,(M-k)*k);beta[coef(fitw,s=lambda_list[j]*sqrt(N/(N-cv.n)))@i]=coef(fitw,s=lambda_list[j]*sqrt(N/(N-cv.n)))@x
      w=matrix(beta,M-k,k)
      test.set=(1:N)[-(i:(N-cv.n-1+i))]
      loss[j]=loss[j]+sum((X[tset,test.set]-t(w)%*%X[-tset,test.set])^2)
    }
  }
  lambda=lambda_list[which.min(loss)]
  #Fit the whole data set
  Z=matrix(rep(0,N*(M-k)*k^2),k*N,k*(M-k))
  for(i in 1:k)
  {
    Z[(N*(i-1)+1):(N*i),((M-k)*(i-1)+1):((M-k)*i)]=t(X)[1:N,-tset]
  }
  Y=as.vector(t(X)[1:N,tset])
  fitw=glmnet(Z,Y,lambda=lambda,intercept=F)
  beta=rep(0,(M-k)*k);beta[coef(fitw)@i]=coef(fitw)@x
  w=matrix(beta,M-k,k)
  return(w)
}

est.wlist=function(M,N,X,test,C_list)
{
  #For each row specified in test, estimate the corresponding w for the tested entries in that row
  w.list=list()
  for(k in 1:length(test))
  {
    tset=test[[k]][-1]
    w.list[[k]]=est.w(M,N,length(tset),X,tset,C_list)
  }
  return(w.list)
}

est.S=function(M,N,X,d,test,w.list)
{
  #Estimate the covariance matrix of the test statistic, 
  #where d is the number of parameters to be tested. 
  #Two estimators invS (symmetric) and CR_invS (asymmetric) are considered.
  invS=matrix(rep(0,d^2),d,d)
  CR_invS=matrix(rep(0,d^2),d,d)
  n1=1
  for(n in 1:length(test))
  {
    tset=test[[n]][-1]
    n2=length(tset)+n1-1
    sym_S=(X[tset,1:N]-t(w.list[[n]])%*%X[-tset,1:N])%*%
      t(X[tset,1:N]-t(w.list[[n]])%*%X[-tset,1:N])/N
    invS[(n1:n2),(n1:n2)]=solve(sym_S)
    asym_invS=solve((X[tset,1:N]-t(w.list[[n]])%*%X[-tset,1:N])%*%t(X)[1:N,tset]/N)
    CR_invS[(n1:n2),(n1:n2)]=t(asym_invS)%*%sym_S%*%asym_invS
    n1=n2+1
  }
  return(list(invS,CR_invS))
}

est.var=function(M,N,r,X,Ah,row)
{
  #Estimate the noise variance based on estimators for some rows of A
  var=sum((X[row,2:(N+1)]-Ah%*%X[,1:N])^2)/r/N
  return(var)
}

test_statistic=function(M,N,X,test,mu)
{
  #Calculate the test statistic for the hypothesis: the entries of A (indicated in test) to be mu
  #Output two test statistics with different estimates of covariance matrix
  
  #estimate A
  r=length(test)
  row=NULL;d=0;
  for(i in 1:r){
    row=c(row,test[[i]][1])
    d=d+length(test[[i]])-1
  }
  Ah=est.A(M,N,r,X,row,seq(0.0005,0.02,0.0005)*7*sqrt(2000/log(20)))
  #w.list
  w.list=est.wlist(M,N,X,test,seq(0.0005,0.02,0.0005)*7*sqrt(2000/log(20)))
  #invS
  out=est.S(M,N,X,d,test,w.list)
  invS=out[[1]]
  CR_invS=out[[2]]
  #var
  var=est.var(M,N,r,X,Ah,row)
  #d dimensional vector Sh
  Sh=rep(0,d)
  n2=0
  for (k in 1:r)
  {
    n1=n2+1;n2=n2+length(test[[k]])-1
    Sh[n1:n2]=(t(w.list[[k]])%*%X[-test[[k]][-1],1:N]-X[test[[k]][-1],1:N])%*%
      (X[test[[k]][1],2:(N+1)]-as.matrix(t(X)[1:N,test[[k]][-1]])%*%mu[[k]]-t(X[-test[[k]][-1],1:N])%*%Ah[k,-test[[k]][-1]])/N
  }
  chi=rep(0,2)
  chi[1]=N*t(Sh)%*%invS%*%Sh/var
  chi[2]=N*t(Sh)%*%CR_invS%*%Sh/var
  return(chi)
}

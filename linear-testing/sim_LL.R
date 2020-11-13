args = commandArgs(trailingOnly=TRUE);
library(glmnet);library(lpSolve)

#fixed qunatities
Dsize=4;Atypes=8;Msize=4;Nsize=5;Distrnum=2;Expnum=1000
d.list=c(6,4,8,10)
rho.list=c(3,2,4,5,6)
M.list=c(rep(30,Nsize),rep(60,Nsize),rep(120,Nsize),rep(300,Nsize))
N.list=rep(c(100,500,1000,5000,10000),Msize)


#quantities determined by args
i1=floor(args/Expnum/Distrnum/Nsize/Msize/Atypes)+1
d=d.list[i1]
i2=floor(args/Expnum/Distrnum/Nsize/Msize)+1
j=i2-(i1-1)*Atypes #j is for Atype
i3=floor(args/Expnum/Distrnum)+1
k=i3-(i2-1)*Msize*Nsize
M=M.list[k]
N=N.list[k]
i4=floor(args/Expnum)+1
l=i4-(i3-1)*Distrnum
if(l==1)
{
  ErrType="unif";
  tr.var=1/3
}else
{
  ErrType="Gauss"
  tr.var=2
}

set.seed(3682)
if(j==1)
{
  A0=matrix(c(1/4,1/2,1/2,1/4),2,2)
  A=matrix(rep(0,M^2),M,M)
  for(i in 1:(M/nrow(A0)))
  {
    A[c(nrow(A0)*i-1,nrow(A0)*i),c(nrow(A0)*i-1,nrow(A0)*i)]=A0
  }
}else if(j==2)
{
  A=matrix(rep(0,M^2),M,M);A[1,1]=A[1,2]=A[M,M-1]=1/4
  for(i in 2:(M-1))
  {
    A[i,i-1]=A[i,i+1]=1/4
  }
}else if(j==3)
{
  A=matrix(rep(0,M^2),M,M);
  a=matrix(rnorm(9),3,3)
  a=a+t(a)
  a=a*3/4/max(abs(eigen(a)[[1]]))
  for(i in 1:(M/3))
  {
    A[(3*(i-1)+1):(3*i),(3*(i-1)+1):(3*i)]=a
  }
}else #make sure symmetric with row-wise sparsity bounded by rho (possibly some rows are of sparsity smaller than rho)
{
  rho=rho.list[j-3]
  s=rho
  A=matrix(rep(0,M^2),M,M)
  for(i in 1:M)
  {
    if(i==1)
    {
      col=sample((i:M),s)
    }
    else 
    {
      s=rho-sum(A[1:(i-1),i]!=0)
      if(i<=rho)
        col=sample((i:M),s)
      else if(i<M)
      {
        check=apply(A[1:(i-1),i:M]!=0,2,sum)<rho
        if(sum(check)>s){
          col=sample((i:M)[check],s)
        }else{
          col=(i:M)[check]
          s=sum(check)
        }
      }else
      {
        check=(sum(A[1:(M-1),M]!=0)<rho)
        col=c(M)[check]
        s=sum(check)
      }
    }
    A[i,col]=rnorm(s)
  }
  A=A+t(A)-diag(diag(A))
  A=A*3/4/max(abs(eigen(A)[[1]]))
}

set.seed(3682)
if(d==6)
{
  test=list(c(1,3,5),c(3,3,4),c(5,4,8))
}else{
  test=list()
  r0=sample(M,d/2)
  ind=sample(M*d/2,d)
  r=ceiling(ind/M)
  c=ind-(r-1)*M
  selected.r=unique(sort(r))
  for(j in 1:length(selected.r))
    test[[j]]=c(r0[selected.r[j]],c[r==selected.r[j]])
}

mu=list()
for(i in 1:length(test))
{
  m=test[[i]][1];
  tset=test[[i]][-1]
  mu[[i]]=A[m,tset]
}

#functions needed for chi_distr
data.generate=function(M,N,A,ErrType,tr.var,Sigma)
{
  if(ErrType=="unif")
  {
    X=runif(M,min=-1,max=1)*sqrt(3*tr.var)
    for(i in 1:10000)
    {
      X=A%*%X+runif(M,min=-1,max=1)*sqrt(3*tr.var)
    }
    for(t in 1:N)
    {
      X=cbind(X,A%*%X[,t]+runif(M,min=-1,max=1)*sqrt(3*tr.var))
    }
  }
  else
  {
    L=svd(Sigma);sqrtS=(L$u)%*%diag(sqrt(L$d))%*%t(L$v)
    X=sqrtS%*%rnorm(M)
    for(t in 1:N)
    {
      X=cbind(X,A%*%X[,t]+rnorm(M,sd=sqrt(tr.var)))
    }
  }
  return(X)
}

est.AL=function(M,N,r,X,A,row)
{
  #devide s by r since the dimension of Y is rN
  s=seq(0.0005,0.02,0.0005)*7*sqrt(log(M)*2000/log(20)/N)/r
  loss=rep(0,length(s))
  cv.n=N/10
  for(i in 1:cv.n)
  {
    Z=matrix(rep(0,M*(N-cv.n)*r^2),r*(N-cv.n),r*M)
    for(k in 1:r)
    {
      Z[((N-cv.n)*(k-1)+1):((N-cv.n)*k),(M*(k-1)+1):(M*k)]=t(X)[i:(N-cv.n-1+i),]
    }
    Y=as.vector(t(X)[(i+1):(N-cv.n+i),row])
    fitA=glmnet(Z,Y,lambda=s*sqrt(N/(N-cv.n)),intercept=F)
    for(j in 1:length(s))
    {
      beta=rep(0,M*r);beta[coef(fitA,s=s[j]*sqrt(N/(N-cv.n)))@i]=coef(fitA,s=s[j]*sqrt(N/(N-cv.n)))@x
      Ah=t(matrix(beta,M,r))
      test.set=(1:N)[-(i:(N-cv.n-1+i))]
      loss[j]=loss[j]+sum((X[row,test.set+1]-Ah%*%X[,test.set])^2)
    }
  }
  lambda=s[which.min(loss)]
  Z=matrix(rep(0,M*N*r^2),r*N,r*M)
  for(i in 1:r)
  {
    Z[(N*(i-1)+1):(N*i),(M*(i-1)+1):(M*i)]=t(X)[1:N,]
  }
  Y=as.vector(t(X)[2:(N+1),row])
  fitA=glmnet(Z,Y,lambda=lambda,intercept=F)
  beta=rep(0,M*r);beta[coef(fitA)@i]=coef(fitA)@x
  Ah=t(matrix(beta,M,r))
  return(list(Ah,sum((Ah-A[row,])^2)))
}


est.wL=function(M,N,k,X,tset)
{
  s=seq(0.0005,0.02,0.0005)*7*sqrt(log(M)*2000/log(20)/N)/k
  loss=rep(0,length(s))
  cv.n=N/10
  for(i in 1:cv.n)
  {
    Z=matrix(rep(0,(N-cv.n)*(M-k)*k^2),k*(N-cv.n),k*(M-k))
    for(n in 1:k)
    {
      Z[((N-cv.n)*(n-1)+1):((N-cv.n)*n),((M-k)*(n-1)+1):((M-k)*n)]=t(X)[i:(N-cv.n-1+i),-tset]
    }
    Y=as.vector(t(X)[i:(N-cv.n-1+i),tset])
    fitw=glmnet(Z,Y,lambda=s*sqrt(N/(N-cv.n)),intercept=F)
    for(j in 1:length(s))
    {
      beta=rep(0,(M-k)*k);beta[coef(fitw,s=s[j]*sqrt(N/(N-cv.n)))@i]=coef(fitw,s=s[j]*sqrt(N/(N-cv.n)))@x
      w=matrix(beta,M-k,k)
      test.set=(1:N)[-(i:(N-cv.n-1+i))]
      loss[j]=loss[j]+sum((X[tset,test.set]-t(w)%*%X[-tset,test.set])^2)
    }
  }
  lambda=s[which.min(loss)]
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

est.wlist=function(M,N,X,test,tr.w)
{
  w.list=list()
  e=rep(0,length(test))
  for(k in 1:length(test))
  {
    tset=test[[k]][-1]
    w.list[[k]]=est.wL(M,N,length(tset),X,tset)
    e[k]=sum((w.list[[k]]-tr.w[[k]])^2)
  }
  return(list(w.list,e))
}

est.S=function(M,N,X,d,test,w.list,tr.invS)
{
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
  return(list(invS,CR_invS,sum((invS-tr.invS)^2),sum((CR_invS-tr.invS)^2)))
}

est.var=function(M,N,r,X,Ah,row,tr.var)
{
  var=sum((X[row,2:(N+1)]-Ah%*%X[,1:N])^2)/r/N
  #sigma squared
  CR_var=sum(diag((X[row,2:(N+1)]-Ah%*%X[,1:N])%*%t(X[row,2:(N+1)])))/r/N
  return(list(var,CR_var,var/tr.var,CR_var/tr.var))
}

chi_distr=function(M,N,A,test,mu,ErrType,tr.var)
{
  #calculate useful quantities
  r=length(test)
  row=NULL;d=0
  for(k in 1:r)
  {
    row=c(row,test[[k]][1])
    d=d+length(test[[k]])-1
  }
  theta=(diag(M)-A%*%A)/tr.var
  Sigma=solve(theta)
  tr.w=list();tr.invS=matrix(rep(0,d^2),d,d)
  n1=1
  for(n in 1:r)
  {
    tr.w[[n]]=-theta[-test[[n]][-1],test[[n]][-1]]%*%solve(theta[test[[n]][-1],test[[n]][-1]])
    tset=test[[n]][-1]
    n2=length(tset)+n1-1
    tr.invS[(n1:n2),(n1:n2)]=theta[tset,tset]
    n1=n2+1
  }
  
  #data generation
  X=data.generate(M,N,A,ErrType,tr.var,Sigma)
  #Ah
  out=est.AL(M,N,r,X,A,row)
  Ah=out[[1]]
  error=out[[2]]
  #w.list
  out=est.wlist(M,N,X,test,tr.w)
  w.list=out[[1]]
  error=c(error,out[[2]])
  #invS
  out=est.S(M,N,X,d,test,w.list,tr.invS)
  invS=out[[1]]
  CR_invS=out[[2]]
  error=c(error,out[[3]],out[[4]])
  #var
  out=est.var(M,N,r,X,Ah,row,tr.var)
  var=out[[1]]
  CR_var=out[[2]]
  error=c(error,out[[3]],out[[4]])
  
  #d dimensional vector Sh
  Sh=rep(0,d)
  n2=0
  for (k in 1:r)
  {
    n1=n2+1;n2=n2+length(test[[k]])-1
    Sh[n1:n2]=(t(w.list[[k]])%*%X[-test[[k]][-1],1:N]-X[test[[k]][-1],1:N])%*%
    (X[test[[k]][1],2:(N+1)]-as.matrix(t(X)[1:N,test[[k]][-1]])%*%mu[[k]]-t(X[-test[[k]][-1],1:N])%*%Ah[k,-test[[k]][-1]])/N
  }
  chi=rep(0,6)
  chi[1]=N*t(Sh)%*%invS%*%Sh/var
  chi[2]=N*t(Sh)%*%CR_invS%*%Sh/var
  chi[3]=N*t(Sh)%*%invS%*%Sh/CR_var
  chi[4]=N*t(Sh)%*%invS%*%Sh/(CR_var)^2*var
  chi[5]=N*t(Sh)%*%CR_invS%*%Sh/CR_var
  chi[6]=N*t(Sh)%*%CR_invS%*%Sh/(CR_var)^2*var
  
  #CR size
  eig=prod(eigen(invS)[[1]])
  return(list(chi,error,eig))
}

set.seed(args)
ptm=proc.time()
out=chi_distr(M,N,A,test,mu,ErrType,tr.var)
time=proc.time()-ptm
save(out,time,file=paste("sim_LL",args-160000,".RData",sep=""))
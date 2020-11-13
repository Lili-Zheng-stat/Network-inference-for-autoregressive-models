function [grad_A, grad_nu]=calc_grad_MN(A,nu,X)
%calculate multinomial gradient w.r.t A and nu with data X. 

[T,M,K]=size(X);
T=T-1;
covariate=repmat(reshape(X(1:T,:,:),T,1,1,M,K),1,M,K,1,1);
coef=repmat(reshape(A,1,M,K,M,K),T,1,1,1,1);
param=sum(sum(covariate.*coef,5),4)+repmat(reshape(nu,1,M,K),T,1,1);
response=X(2:(T+1),:,:);
deviation=exp(param)./repmat(1+sum(exp(param),3),1,1,K)-response;
grad_nu=reshape(sum(deviation,1)/T,M,K);
grad_A=reshape(sum(repmat(reshape(deviation,T,M,K,1,1),1,1,1,M,K).*covariate,1)/T,M,K,M,K);
end

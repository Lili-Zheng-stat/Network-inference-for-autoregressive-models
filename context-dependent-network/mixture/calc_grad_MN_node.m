function [grad_A, grad_nu]=calc_grad_MN_node(A,nu,X_cov,X_resp)
%calculate multinomial gradient w.r.t A (1*K*M*K) and nu (1*K) with covariates X_cov of size T*M*K, response X_resp of size T*K. 

[T,M,K]=size(X_cov);
covariate=repmat(reshape(X_cov,T,1,M,K),1,K,1,1);
coef=repmat(reshape(A,1,K,M,K),T,1,1,1);
param=sum(sum(covariate.*coef,4),3)+repmat(nu,T,1);
deviation=exp(param)./repmat(1+sum(exp(param),2),1,K)-reshape(X_resp,T,K);
grad_nu=mean(deviation,1);
grad_A=reshape(sum(repmat(reshape(deviation,T,K,1,1),1,1,M,K).*covariate,1)/T,1,K,M,K);
end
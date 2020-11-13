function [grad_A, grad_nu]=calc_grad_LN_node(A,nu,X_cov,X_resp)
%calculate gradient w.r.t A (1*K*M*K) and nu (1*K) with covariates X_cov of size T*M*K and response X_resp of size T*K.

[T,M,K]=size(X_cov);
grad_A=zeros(1,K,M,K);
grad_nu=zeros(1,K);
covariate=reshape(X_cov,T,1,M*K);
coef=repmat(reshape(A(1,K,:),1,1,M*K),T,1,1);
intcpt=ones(T,1)*nu(1,K);
p_temp=reshape(1./(1+exp(-sum(covariate.*coef,3)-intcpt)),T,1);
ind_event=reshape(double(sum(X_resp,3)~=0),T,1);
disc_dev=repmat(reshape(p_temp-ind_event,T,1,1,1),1,1,M,K);
past_covariate=reshape(X_cov,T,1,M,K);
grad_A(1,K,:,:)=reshape(mean(disc_dev.*past_covariate,1),1,1,M,K);
grad_nu(1,K)=mean(p_temp-ind_event,1);
for i=1:(K-1)
    Y=reshape(log(X_resp(:,1,i)./X_resp(:,1,K)),T,1);
    Y(find(ind_event==0))=0;
    coef_cts=repmat(reshape(A(1,i,:),1,1,M*K),T,1,1);
    mu=sum(covariate.*coef_cts,3)+ones(T,1)*nu(1,i);
    cts_dev=repmat(reshape((Y-mu).*ind_event,T,1,1,1),1,1,M,K);
    grad_A(1,i,:,:)=reshape(-sum(cts_dev.*past_covariate,1)/T,1,M,K);
    grad_nu(1,i)=-mean((Y-mu).*ind_event,1);
end
end
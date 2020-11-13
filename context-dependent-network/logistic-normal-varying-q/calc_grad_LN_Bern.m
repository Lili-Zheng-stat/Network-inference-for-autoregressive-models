function [grad_A, grad_nu]=calc_grad_LN_Bern(A,nu,X)
%calculate gradient w.r.t A and nu with data X under the logistic-normal
%model with time-varying q.

[T,M,K]=size(X);
T=T-1;
grad_A=zeros(M,K,M,K);
grad_nu=zeros(M,K);
covariate=repmat(reshape(X(1:T,:),T,1,M*K),1,M,1);
coef=repmat(reshape(A(:,K,:),1,M,M*K),T,1,1);
intcpt=repmat(reshape(nu(:,K),1,M),T,1);
p_temp=reshape(1./(1+exp(-sum(covariate.*coef,3)-intcpt)),T,M);
ind_event=reshape(double(sum(reshape(X(2:(T+1),:,:),T,M,K),3)~=0),T,M);
disc_dev=repmat(reshape(p_temp-ind_event,T,M,1,1),1,1,M,K);
past_covariate=repmat(reshape(X(1:T,:,:),T,1,M,K),1,M,1,1);
grad_A(:,K,:,:)=reshape(mean(disc_dev.*past_covariate,1),M,M,K);
grad_nu(:,K)=reshape(mean(reshape(p_temp-ind_event,T,M),1),M,1);
for i=1:(K-1)
    Y=reshape(log(X(2:(T+1),:,i)./X(2:(T+1),:,K)),T,M);
    Y(find(ind_event==0))=0;
    coef_cts=repmat(reshape(A(:,i,:),1,M,M*K),T,1,1);
    mu=sum(covariate.*coef_cts,3)+repmat(reshape(nu(:,i),1,M),T,1);
    cts_dev=repmat(reshape((Y-mu).*ind_event,T,M,1,1),1,1,M,K);
    grad_A(:,i,:,:)=reshape(-sum(cts_dev.*past_covariate,1)/T,M,M,K);
    grad_nu(:,i)=reshape(-mean(reshape((Y-mu).*ind_event,T,M),1),M,1);
end
end

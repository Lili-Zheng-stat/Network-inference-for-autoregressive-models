function [Ah,nu_h,Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=fit_LN_Bern_catind(X,lambda, ...
intercept,init_A, init_nu, eta,tol,iter)
%Fit the logistic-normal model with context-independent network. Ah_BAR and nu_h_BAR are the 
%estimated network and offset parameters under the Bernoulli model, 
%Ah and nu_h are the corresponding multinomial network parameter and offset parameter. loss is
%a vector containing the penalized Bernoulli log-likeliihood loss evaluated at each
%iteration; grad_A and grad_nu are the gradients w.r.t. Ah_BAR and nu_h_BAR when the
%BAR algorithm stops; cvg = true if tolerance is met when the BAR algorithm stops.

[~,M,K]=size(X);
[Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(sum(X,3),lambda,intercept,...
    init_A, init_nu, eta,tol,iter);
Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
Ah(:,K,:,:)=repmat(reshape(Ah_BAR,M,1,M,1),1,1,1,K);
nu_h(:,K)=reshape(nu_h_BAR,M,1);
for m=1:M
    ind_temp=find(sum(X(:,m,:),3)~=0);
    nu_h(m,1:(K-1))=reshape(mean(log(X(ind_temp,m,1:(K-1))./X(ind_temp,m,K)),1),1,K-1);
end

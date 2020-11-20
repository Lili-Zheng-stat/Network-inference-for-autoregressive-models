function [Ah,nu_h,Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=fit_MN_catind(X,lambda, ...
intercept,init_A, init_nu, eta,tol,iter)
%Fit the multinomial model with context-independent network. Ah_BAR and nu_h_BAR are the 
%estimated network and offset parameters under the Bernoulli model, 
%Ah and nu_h are the corresponding multinomial network parameter and offset parameter. loss is
%a vector containing the penalized Bernoulli log-likeliihood loss evaluated at each
%iteration; grad_A and grad_nu are the gradients w.r.t. Ah_BAR and nu_h_BAR when the
%BAR algorithm stops; cvg = true if tolerance is met when the BAR algorithm stops.

[~,M,K]=size(X);
[Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(sum(X,3),lambda,intercept,...
    init_A, init_nu, eta,tol,iter);
Ah=repmat(reshape(Ah_BAR,M,1,M,1),1,K,1,K);
p_hat=zeros(M,K);
for m=1:M
    ind_temp=find(sum(X(:,m,:),3)~=0);
    p_hat(m,:)=mean(X(ind_temp,m,:),1);
end
nu_h=repmat(nu_h_BAR,1,K)+log(p_hat);
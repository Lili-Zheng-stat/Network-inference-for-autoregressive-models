function [ah,sigma_MN_sq_hat]=est_a_sigma(X_cov,X_resp,Ah_MN,nu_h_MN)
%estimate the Gaussian mixture parameters a and sigma squared for each node, given
%estimated A (M*K*M*K) and nu (M*K) under the multinomial model.

[T,M,K]=size(X_cov);
ah=zeros(1,M);sigma_MN_sq_hat=zeros(1,M);
for m=1:M
    nevent=0;logratio_sq_sum=0;
    mixt_sq_expt_sum=0;ah_numer=0;ah_denom=0;
    for t=1:T
        if sum(X_resp(t,m,:),3)>0
            nevent=nevent+1;
            logratio=reshape(log(X_resp(t,m,1:(K-1))/X_resp(t,m,K)),1,K-1);
            logratio_sq_sum=logratio_sq_sum+sum(logratio.^2);
            param=zeros(1,K);
            for k=1:K
                param(k)=sum(reshape(Ah_MN(m,k,:,:),1,M*K).*reshape(X_cov(t,:,:),1,M*K))+nu_h_MN(m,k);
            end
            mixt_expt=(exp(param(1:(K-1)))-exp(param(K)))/sum(exp(param));
            ah_numer=ah_numer+sum(logratio.*mixt_expt);
            ah_denom=ah_denom+sum(mixt_expt.^2);
            mixt_sq_expt_sum=mixt_sq_expt_sum+(sum(exp(param(1:(K-1))))+(K-1)*exp(param(K)))/sum(exp(param));
        end
    end
    ah(m)=ah_numer/ah_denom;
    logratio_sq_mean=logratio_sq_sum/nevent;
    mixt_sq_expt_mean=mixt_sq_expt_sum/nevent;
    sigma_MN_sq_hat(m)=max((logratio_sq_mean-mixt_sq_expt_mean*ah(m)^2)/(K-1),0.001);
end

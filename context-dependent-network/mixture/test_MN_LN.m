function [logL_LN_est,logL_MN_est]=test_MN_LN(X_cov,X_resp,Ah_MN,nu_h_MN,a,sigma_sq,Ah_LN,nu_h_LN)
%log likelihood function for each node under two estimated models

[T,M,K]=size(X_cov);
logL_LN_est=zeros(1,M);logL_MN_est=zeros(1,M);
for m=1:M
%likelihood ratio for node m
    nevent=0;est_cov=zeros(K-1,K-1);
    for t=1:T
    %likelihood of MN and LN model evaluated at time t+1
        param_MN=zeros(1,K);param_LN=zeros(1,K);
        for k=1:K
            param_MN(k)=sum(reshape(Ah_MN(m,k,:,:),1,M*K).*reshape(X_cov(t,:,:),1,M*K))+nu_h_MN(m,k);
            param_LN(k)=sum(reshape(Ah_LN(m,k,:,:),1,M*K).*reshape(X_cov(t,:,:),1,M*K))+nu_h_LN(m,k);
        end
        if sum(X_resp(t,m,:),3)>0
            logratio=reshape(log(X_resp(t,m,1:(K-1))/X_resp(t,m,K)),1,K-1);
            density=zeros(1,K);
            for k=1:(K-1)
                density(k)=exp(-((sum(logratio.^2)+a(m)^2-2*a(m)*logratio(k))/2/sigma_sq(m)))/((2*pi*sigma_sq(m))^((K-1)/2));
            end
            density(K)=exp(-((sum(logratio.^2)+a(m)^2*k+2*a(m)*sum(logratio))/2/sigma_sq(m)))/((2*pi*sigma_sq(m))^((K-1)/2));
            logL_MN_temp=-log(sum(exp(param_MN)/(1+sum(exp(param_MN))).*density));
            logL_LN_temp=log(1+exp(param_LN(K)))-param_LN(K)+(K-1)/2*log(2*pi);
            err_temp=logratio-param_LN(1:(K-1));
            est_cov=est_cov+err_temp'*err_temp;
            nevent=nevent+1;
        else
            logL_MN_temp=log(1+sum(exp(param_MN)));
            logL_LN_temp=log(1+exp(param_LN(K)));
        end
        logL_MN_est(m)=logL_MN_est(m)+logL_MN_temp;
        logL_LN_est(m)=logL_LN_est(m)+logL_LN_temp;
    end
    est_cov=est_cov/nevent;
    logL_LN_est(m)=logL_LN_est(m)+(log(det(est_cov))+K-1)*nevent/2;
end
logL_LN_est=logL_LN_est/T;logL_MN_est=logL_MN_est/T;
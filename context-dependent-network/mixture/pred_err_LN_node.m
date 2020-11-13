function pred_err=pred_err_LN_node(X_cov,X_resp,Ah_LN,nu_h_LN)
%find out the prediction error of A (1*K*M*K), nu (1*K) on data X_cov (T*M*K), X_resp (T*1*K), under the
%logistic-normal model

[T,M,K]=size(X_cov);
pred_err=0;
for t=1:T
    param=zeros(1,K);
    for k=1:K
        param(k)=sum(reshape(Ah_LN(1,k,:,:),1,M*K).*reshape(X_cov(t,:,:),1,M*K))+nu_h_LN(1,k);
    end
    prob=exp(param(K))/(1+exp(param(K)));
    expt_logratio=param(1:(K-1));
    pred=zeros(1,K);
    pred(1:(K-1))=exp(expt_logratio)/(1+sum(exp(expt_logratio)));
    pred(K)=1/(1+sum(exp(expt_logratio)));
    pred=prob*pred;
    pred_err=pred_err+sum((pred-reshape(X_resp(t,1,:),1,K)).^2);
end
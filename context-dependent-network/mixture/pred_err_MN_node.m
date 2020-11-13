function [pct_err,prob_err]=pred_err_MN_node(X_cov,X_resp,A,nu)
%find out the prediction error (average number of wrong prediction per node) 
%of A (1*K*M*K), nu (1*K) on data X_cov (T*M*K), X_resp (T*1*K), under the
%multinomial model

[T,M,K]=size(X_cov);
pct_err=0;prob_err=0;
for t=1:T
    mu=ones(1,K+1);
    for i=1:K
        mu(i)=exp(sum(reshape(A(1,i,:),1,M*K).*X_cov(t,:))+nu(1,i));
    end
    [~,I_pred]=max(mu);
    if sum(X_resp(t,1,:))==0
       I_truth=K+1;
    else
       [~,I_truth]=max(X_resp(t,1,:));
    end
    pct_err=pct_err+double(I_pred~=I_truth);
    prob_err=prob_err+sum((mu(1:K)/sum(mu)-reshape(X_resp(t,1,:),1,K)).^2);
end
end
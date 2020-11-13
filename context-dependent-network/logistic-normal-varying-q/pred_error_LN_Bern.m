function pred_err=pred_error_LN_Bern(X,A,nu)
%find out the prediction error of A, nu on data X, under the
%logistic-normal model with time-varying q
[T,M,K]=size(X);
T=T-1;
pred_err=0;
for t=1:T
    for m=1:M
        pred_q=1/(1+exp(-sum(reshape(A(m,K,:,:),1,M*K).*reshape(X(t,:,:),1,M*K))-nu(m,K)));
        mu=ones(1,K);
        for i=1:(K-1)
            mu(i)=exp(sum(reshape(A(m,i,:),1,M*K).*X(t,:))+nu(m,i));
        end
        pred=pred_q*mu/sum(mu);
        pred_err=pred_err+sum((pred-reshape(X(t+1,m,:),1,K)).^2);
    end
end
pred_err=pred_err/M;
end
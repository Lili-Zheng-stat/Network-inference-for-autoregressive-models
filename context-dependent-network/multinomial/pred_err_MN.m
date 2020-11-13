function pct_err=pred_err_MN(X,A,nu)
%find out the prediction error (average number of wrong prediction per node,
%summed over time points) of A, nu on data X
[T,M,K]=size(X);
T=T-1;
pct_err=0;
for t=1:T
    for m=1:M
        mu=ones(1,K+1);
        for i=1:K
            mu(i)=exp(sum(reshape(A(m,i,:),1,M*K).*X(t,:))+nu(m,i));
        end
        [~,I_pred]=max(mu);
        if sum(X(t+1,m,:))==0
            I_truth=K+1;
        else
            [~,I_truth]=max(X(t+1,m,:));
        end
        pct_err=pct_err+double(I_pred~=I_truth);
    end
end
pct_err=pct_err/M;
end
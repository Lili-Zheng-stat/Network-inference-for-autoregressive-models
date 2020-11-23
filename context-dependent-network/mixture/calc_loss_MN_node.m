function loss=calc_loss_MN_node(A,nu,X_cov,X_resp)
%calculte the multinomial log-likelihood loss function at A (1*K*M*p), intercept nu (1*K),
%and covariate X_cov, response X_resp

[T,M,K]=size(X_cov);
term1=zeros(T,1);term2=0;
covariate=reshape(X_cov,[T 1 M K]);
for i=1:K
    coef=repmat(A(1,i,:,:),T,1,1,1);
    param=sum(sum(covariate.*coef,4),3)+ones(T,1)*nu(1,i);
    response=reshape(X_resp(:,1,i),T,1);
    term1=term1+exp(param);
    term2=term2+sum(param(:).*response(:));
end
loss=(sum(log(1+term1(:)))-term2)/T;
end
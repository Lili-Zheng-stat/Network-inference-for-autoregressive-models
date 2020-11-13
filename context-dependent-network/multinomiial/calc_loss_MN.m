function loss=calc_loss_MN(A,nu,X)
%calculte the multinomial log-likelihood loss function at A (M*K*M*K), intercept nu (M*K),
%and data X. 

[T,M,K]=size(X);
T=T-1;
term1=zeros(T,M);term2=0;
covariate=repmat(reshape(X(1:T,:,:),[T 1 M K]),1,M,1,1);
for i=1:K
    coef=repmat(reshape(A(:,i,:,:),1,M,M,K),T,1,1,1);
    param=sum(sum(covariate.*coef,4),3)+repmat(reshape(nu(:,i),1,M),T,1);
    response=reshape(X(2:(T+1),:,i),T,M);
    term1=term1+exp(param);
    term2=term2+sum(param(:).*response(:));
end
loss=(sum(log(1+term1(:)))-term2)/T;
end
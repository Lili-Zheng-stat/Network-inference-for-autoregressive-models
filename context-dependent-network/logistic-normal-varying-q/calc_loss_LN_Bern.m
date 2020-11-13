function loss=calc_loss_LN_Bern(A,nu,X,alpha)
%calculte the log-likelihood loss function at A and nu,
%with data X. alpha is the weight of density log-likelihood

[T,M,K]=size(X);
T=T-1;
disc_coef=repmat(reshape(A(:,K,:),1,M,M*K),T,1,1);
covariate=repmat(reshape(X(1:T,:),T,1,M*K),1,M,1);
param=sum(disc_coef.*covariate,3)+repmat(reshape(nu(:,K),1,M),T,1);
ind_event=double(sum(X(2:(T+1),:,:),3)>0);
disc_loss_mat=log(1+exp(param))-param.*ind_event;
loss=sum(disc_loss_mat(:))/T*(1-alpha);
for i=1:(K-1)
    Y=reshape(log(X(2:(T+1),:,i)./X(2:(T+1),:,K)),T,M);
    Y(find(ind_event==0))=0;
    cts_coef=repmat(reshape(A(:,i,:),1,M,M*K),T,1,1);
    mu=sum(cts_coef.*covariate,3)+repmat(reshape(nu(:,i),1,M),T,1);
    cts_loss_mat=((Y-mu).^2).*ind_event;
    loss=loss+sum(cts_loss_mat(:))/2/T*alpha;
end
end
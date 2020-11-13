function loss=calc_loss_LN_node(A,nu,X_cov,X_resp,alpha)
%calculte the logistic-normal log-likelihood loss function at A (1*K*M*K), intercept nu (1*K),
%and covariates X_cov, response X_resp. alpha is the
%weight of density log-likelihood

[T,M,K]=size(X_cov);
disc_coef=repmat(reshape(A(1,K,:),1,1,M*K),T,1,1);
covariate=reshape(X_cov,T,1,M*K);
param=sum(disc_coef.*covariate,3)+ones(T,1)*nu(1,K);
ind_event=double(sum(X_resp,3)>0);
disc_loss_mat=log(1+exp(param))-param.*ind_event;
loss=sum(disc_loss_mat(:))/T*(1-alpha);
for i=1:(K-1)
    Y=reshape(log(X_resp(:,1,i)./X_resp(:,1,K)),T,1);
    Y(find(ind_event==0))=0;
    cts_coef=repmat(reshape(A(1,i,:),1,1,M*K),T,1,1);
    mu=sum(cts_coef.*covariate,3)+ones(T,1)*nu(1,i);
    cts_loss_mat=((Y-mu).^2).*ind_event;
    loss=loss+sum(cts_loss_mat(:))/2/T*alpha;
end
end
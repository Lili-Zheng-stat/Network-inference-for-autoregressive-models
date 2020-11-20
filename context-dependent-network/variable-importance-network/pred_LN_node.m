function pred=pred_LN_node(Q_cov,Ah_LN,nu_h_LN)
%give prediction for the node in each category at each time t (T*p prediction matrix)
[T,M,p]=size(Q_cov);
pred=zeros(T,p);
param=zeros(T,p);
for k=1:p
    param(:,k)=sum(repmat(reshape(Ah_LN(1,k,:,:),1,M*p),T,1).*reshape(Q_cov,T,M*p),2)+repmat(nu_h_LN(1,k),T,1);
end
prob=exp(param(:,p))./(1+exp(param(:,p)));
expt_logratio=param(:,1:(p-1));
pred(:,1:(p-1))=exp(expt_logratio)./repmat(1+sum(exp(expt_logratio),2),1,p-1);
pred(:,p)=1/(1+sum(exp(expt_logratio),2));
pred=repmat(prob,1,p).*pred;
end
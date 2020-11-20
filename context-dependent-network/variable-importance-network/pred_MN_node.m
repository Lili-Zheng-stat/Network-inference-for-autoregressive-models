function pred=pred_MN_node(Q_cov,Ah_MN,nu_h_MN)
%give prediction for the node in each category at each time t (T*p prediction matrix)
[T,M,p]=size(Q_cov);
param=zeros(T,p);
for k=1:p
    param(:,k)=sum(repmat(reshape(Ah_MN(1,k,:,:),1,M*p),T,1).*reshape(Q_cov,T,M*p),2)+repmat(nu_h_MN(1,k),T,1);
end
pred=exp(param)./repmat((1+sum(exp(param),2)),1,p);

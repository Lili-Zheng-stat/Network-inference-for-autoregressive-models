function Infl_network=var_importance(X_cov,A,nu,LN_list)
[T,M,K]=size(X_cov);
Infl_network=zeros(M,K,M,K);
for m1=1:M
    if sum(LN_list==m1)>0
        pred_current=pred_LN_node(X_cov,A(m1,:,:,:),nu(m1,:));
    else
        pred_current=pred_MN_node(X_cov,A(m1,:,:,:),nu(m1,:));
    end
    pred_mean=zeros(T,K,M,K);sign_dev=zeros(T,1,M,K);
    for m2=1:M
        for k=1:p
            X_mean=X_cov;
            X_mean(:,m2,k)=mean(X_cov(:,m2,k),1);
            if sum(LN_list==m1)>0
                pred_mean(:,:,m2,k)=reshape(pred_LN_node(X_mean,A(m1,:,:,:),nu(m1,:)),T,K,1,1);
            else
                pred_mean(:,:,m2,k)=reshape(pred_MN_node(X_mean,A(m1,:,:,:),nu(m1,:)),T,K,1,1);
            end
            sign_dev(:,1,m2,k)=reshape(X_cov(:,m2,k)>X_mean(:,m2,k),T,1,1,1);
        end
    end
    Infl_network(m1,:,:,:)=mean((repmat(reshape(pred_current,T,K,1,1),1,1,M,K)-pred_mean).*repmat(sign_dev,1,K,1,1),1);
end
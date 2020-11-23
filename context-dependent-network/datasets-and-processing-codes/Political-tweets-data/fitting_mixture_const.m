clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;
load('test_statistic.mat');
LN_list=find(L_diff==-Inf);MN_list=find(L_diff~=-Inf);
    
X_mixed=X_rounded;
for i=1:length(LN_list)
    X_mixed(:,LN_list(i),:)=X(:,LN_list(i),:);
end
X_mixed_train=X_mixed(1:T_train,:,:);X_mixed_test=X_mixed((T_train+1):(T+1),:,:);

%training and prediction for a constant mixture process
pred_err=zeros(M,T_test-1);
for m=1:M
    if (sum(LN_list==m)>0)
        nu_constant=zeros(1,K);
        ratio_inverse=T_train/sum(sum(X_mixed_train(:,m,:),3)>0);
        if ratio_inverse>1
            nu_constant(m,K)=-log(ratio_inverse-1);
        else 
            nu_constant(m,K)=50;
        end
        ind_temp=find(sum(X_mixed_train(:,m,:),3)~=0);
        nu_constant(m,1:(K-1))=reshape(mean(log(X_mixed_train(ind_temp,m,1:(K-1))./...
         X_mixed_train(ind_temp,m,K)),1),1,K-1);
        pred_err(m,:)=pred_err_LN_node(X_mixed_test(1:(T_test-1),:,:),X_mixed_test(2:T_test,m,:),zeros(1,K,M,K),nu_constant);
    else
        p_hat_constant=reshape(mean(X_mixed_train(:,m,:),1),1,K);
        [~,I_pred]=max([p_hat_constant 1-sum(p_hat_constant,2)],[],2);
        I_truth=zeros(T_test-1,1);
        for t=1:(T_test-1)
            if sum(X_mixed_test(t+1,m,:))==0
                I_truth(t,1)=K+1;
            else
                [~,I_truth(t,1)]=max(X_mixed_test(t+1,m,:));
            end
            pred_err(m,:)=double(I_truth~=repmat(I_pred,T_test-1,1));
        end
    end
end
mean(mean(pred_err,1))
%0.293554151771677

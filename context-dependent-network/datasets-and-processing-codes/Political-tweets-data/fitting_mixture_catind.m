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

%cross validation for category independent mixture network
lambda_c_list=0.01:0.01:0.1;
pred_err=zeros(1,length(lambda_list));
for i=1:length(lambda_list)
    output=cv_mix_catind(X_mixed_train,LN_list,lambda_c_list(i),true,init_A,init_nu,eta,tol,iter);
    pred_err(i)=mean(cell2mat(output.pred_err)); 
end
[~,I]=min(pred_err);%lambda_c_list(2)
lambda=lambda_c_list(I)*sqrt(log(M)/T_train);

%training and prediction of the mixture model with category-independent
%network
init_A=zeros(M,M);init_nu=zeros(M,1);
eta=2;tol=0.0001;iter=2000;
output=struct;
[Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=...
    fit_BAR(sum(X_mixed_train,3),lambda,true,init_A, init_nu, eta,tol,iter);
Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
for m=1:M
    ind_temp=find(sum(X_mixed_train(:,m,:),3)~=0);
    if sum(LN_list==m)>0
        Ah(m,K,:,:)=repmat(reshape(Ah_BAR(m,:),1,1,M,1),1,1,1,K);
        nu_h(m,K)=nu_h_BAR(m);
        nu_h(m,1:(K-1))=reshape(mean(log(X_mixed_train(ind_temp,m,1:(K-1))./...
            X_mixed_train(ind_temp,m,K)),1),1,K-1);
        pred_err(m,:)=pred_err_LN_node(X_mixed_test(1:(T_test-1),:,:),X_mixed_test(2:T_test,m,:),Ah(m,:,:,:),nu_h(m,:));
    else
        Ah(m,:,:,:)=repmat(reshape(Ah_BAR(m,:),1,1,M,1),1,K,1,K);
        p_hat=mean(X_mixed_train(ind_temp,m,:),1);
        nu_h(m,:)=repmat(nu_h_BAR(m),1,K)+reshape(log(p_hat),1,K);
        [pred_err(m,:),~]=pred_err_MN_node(X_mixed_test(1:(T_test-1),:,:),X_mixed_test(2:T_test,m,:),Ah(m,:,:,:),nu_h(m,:));
    end
end
mean(mean(pred_err,1),2)
%prediction error: 0.229206031082712


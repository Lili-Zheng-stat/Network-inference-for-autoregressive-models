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

%cross validation for the mixture model
lambda_c_MN_list=0.016:0.002:0.04;
lambda_c_LN_list=0.035:0.002:0.059;alpha_list=[0.3 0.5 0.7];

%cv for LN nodes
pred_err_LN=zeros(length(LN_list),length(alpha_list),length(lambda_c_LN_list));
init_A=zeros(1,K,M,K);init_nu=zeros(1,K);
eta=2;tol=0.0001;iter=1000;
for i=1:length(LN_list)
    m=LN_list(i);
    for j1=1:length(alpha_list)
        for j2=1:length(lambda_c_LN_list)
            output=cv_LN_node(X_mixed(1:(T_train-1),:,:),X_mixed(2:T_train,m,:),lambda_c_LN_list(j2),alpha_list(j1),true,init_A,init_nu,eta,tol,iter);
            pred_err_LN(i,j1,j2)=mean(cell2mat(output.pred_err));  
        end
    end
end
[~,ind]=min(reshape(mean(pred_err_LN,1),length(alpha_list)*length(lambda_c_LN_list),1)); 
tuning_choice_LN_lambda=ceil(ind/length(alpha_list));tuning_choice_LN_alpha=ind-(tuning_choice_LN_lambda-1)*length(alpha_list);
LN_alpha=alpha_list(tuning_choice_LN_alpha);LN_lambda_c=lambda_c_LN_list(tuning_choice_LN_lambda);

%cv for MN nodes
pred_err_MN=zeros(length(MN_list),length(lambda_c_MN_list));
for i=1:length(MN_list)
    m=MN_list(i);
    for j=1:length(lambda_c_MN_list)
        output=cv_MN_node(X_mixed(1:(T_train-1),:,:),X_mixed(2:T_train,m,:),lambda_c_MN_list(j),true,init_A,init_nu,eta,tol,iter);
        pred_err_MN(i,j)=mean(cell2mat(output.pct_err));  
    end
end
[~,tuning_choice_MN_lambda]=min(reshape(mean(mean(pred_err_MN,3),1),length(lambda_c_MN_list),1)); 
MN_lambda_c=lambda_c_MN_list(tuning_choice_MN_lambda);

%training and prediction for the mixture model
X_mixed_train=X_mixed(1:T_train,:,:);X_mixed_test=X_mixed((T_train+1):(T+1),:,:);
%prediction error of a context-dependent network
LN_lambda=lambda_c_LN_list(tuning_choice_LN_lambda)*K*sqrt(log(M)/T_train);
MN_lambda=lambda_c_MN_list(tuning_choice_MN_lambda)*K*sqrt(log(M)/T_train);
Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
pred_err=zeros(M,T_test-1);
for m=1:M
    if sum(LN_list==m)>0
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_LN_node(X_mixed_train(1:(T_train-1),:,:),X_mixed_train(2:T_train,m,:),LN_lambda,LN_alpha,true,init_A, init_nu, eta,tol,iter);
        pred_err(m,:)=pred_err_LN_node(X_mixed_test(1:(T_test-1),:,:),X_mixed_test(2:T_test,m,:),Ah(m,:,:,:),nu_h(m,:));
    else
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_MN_node(X_mixed_train(1:(T_train-1),:,:),X_mixed_train(2:T_train,m,:),MN_lambda,true,init_A, init_nu, eta,tol,iter);
        [pred_err(m,:),~]=pred_err_MN_node(X_mixed_test(1:(T_test-1),:,:),X_mixed_test(2:T_test,m,:),Ah(m,:,:,:),nu_h(m,:));
    end
end
mean(mean(pred_err,1))
%0.229134815266037

%fit the whole data using the mixture model
LN_lambda=lambda_c_LN_list(tuning_choice_LN_lambda)*K*sqrt(log(M)/T);
MN_lambda=lambda_c_MN_list(tuning_choice_MN_lambda)*K*sqrt(log(M)/T);
for m=1:M
    if sum(LN_list==m)>0
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_LN_node(X_mixed(1:T,:,:),X_mixed(2:(T+1),m,:),LN_lambda,LN_alpha,true,init_A, init_nu, eta,tol,iter);
    else
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_MN_node(X_mixed(1:T,:,:),X_mixed(2:(T+1),m,:),MN_lambda,true,init_A, init_nu, eta,tol,iter);
    end
end

%transfer the estimated parameters to variable importance networks
Infl_network_mixed=var_importance(X_mixed(1:T,:,:),Ah,nu_h,LN_list);
g_mat_mixed=output_network(Infl_network_mixed,0,true);
csvwrite('twitter_mixed_infl_graph.csv',g_mat_mixed);

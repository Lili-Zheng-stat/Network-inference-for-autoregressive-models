clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%cross validation for the multinomial model
lambda_c_list=(0.005:0.0001:0.0085)/K*sqrt(T_train/log(M));
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);eta=2;tol=0.0001;iter=500;
pred_err_arr=zeros(1,length(lambda_c_list));
for i=1:length(lambda_c_list)
    output=cv_MN(X_train_rounded,lambda_c_list(i),true,init_A,init_nu,eta,tol,iter);
    pred_err_arr(i)=mean(cell2mat(output.pred_err));
end
[~,ind]=min(pred_err_arr);
lambda_c_MN=lambda_c_list(ind);

%training and prediction for the multinomial model
lambda_MN=lambda_c_MN*K*sqrt(log(M)/T_train);
[Ah_MN,nu_h_MN,~,~,~,~]=...
fit_MN(X_train_rounded,lambda_MN,true,init_A, init_nu, eta,tol,iter);
pred_err_MN=pred_err_MN(X_test_rounded,Ah_MN,nu_h_MN)/(T_test-1)
%0.251999418351025

%fit the whole data using the multinomial and logistic-normal (time-varying q) models
lambda_MN=lambda_c_MN*K*sqrt(log(M)/T);
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
eta=2;tol=0.0001;iter=500;
[Ah_MN,nu_h_MN,~,~,~,~]=fit_MN(X_rounded,lambda_MN,true,init_A, init_nu, eta,tol,iter);
save('fitted_MN.mat','Ah_MN','nu_h_MN');

%transfer the estimated parameters to variable importance networks
Infl_network_MN=var_importance(X_rounded(1:T,:,:),Ah_MN,nu_h_MN,[]);
g_mat_MN=output_network(Infl_network_MN,0,true);
csvwrite('twitter_MN_infl_graph.csv',g_mat_MN);




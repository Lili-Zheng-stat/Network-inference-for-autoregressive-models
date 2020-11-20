clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%cross validation for the multinomial model with category-independent network
init_A=zeros(M,M);init_nu=zeros(M,1);
eta=2;tol=0.0001;iter=2000;
lambda_c_list=(0.001:0.001:0.009)/sqrt(log(M)/T_train);
pred_err_arr=zeros(1,length(lambda_c_list));
for i=1:length(lambda_c_list)
    output=cv_MN_catind(X_train_rounded,lambda_c_list(i),true,init_A,init_nu,eta,tol,iter);
    pred_err_arr(i)=mean(cell2mat(output.pred_err));
end
[~,ind]=min(pred_err_arr);
lambda_c_MN_catind=lambda_c_list(ind);

%training and prediction for the multinomial model with
%category-independent network
lambda=lambda_c_MN_catind*sqrt(log(M)/T_train);
[Ah_MN_catind,nu_h_MN_catind,~,~,~,~,~,~]=fit_MN_catind(X_train_rounded,lambda,true,init_A, init_nu, eta,tol,iter);
pred_err_MN_catind=pred_err_MN(X_test_rounded,Ah_MN_catind,nu_h_MN_catind)/(T_test-1)
%0.255198487712665

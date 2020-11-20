clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%cross validation for logistic-normal model with category-independent
%network
init_A=zeros(M,M);init_nu=zeros(M,1);
lambda_c_list=(0.001:0.001:0.009)/sqrt(log(M)/T_train);
eta=2;tol=0.0001;iter=2000;
pred_err_arr=zeros(1,length(lambda_c_list));
for i=1:length(lambda_c_list)
    output=cv_LN_Bern_catind(X_train,lambda_c_list(i),true,init_A,init_nu,eta,tol,iter);
    pred_err_arr(i)=mean(output.pred_err);
end
[~,ind]=min(pred_err_arr);
lambda_c_LN_catind=lambda_c_list(ind);

%training and prediction for the logistic-normal model with
%category-independent network
lambda=lambda_c_LN_catind*sqrt(log(M)/T_train);
[Ah_LN_catind,nu_h_LN_catind,~,~,~,~,~,~]=fit_LN_Bern_catind(X_train,lambda,true,init_A, init_nu, eta,tol,iter);
pred_err_LN_catind=pred_err_LN_Bern(X_test,Ah_LN_catind,nu_h_LN_catind)/(T_test-1)
%prediction error: 0.143732477362186

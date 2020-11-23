clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%testing for the type of each node
%fitting MN and LN models by regressing rounded and original future data
%upon original past data

%cross validation for multinomial model
init_A=zeros(1,K,M,K);init_nu=zeros(1,K);
eta=2;tol=0.0001;iter=1000;
lambda_c_MN_list=0.03:0.001:0.05;
pred_err_arr=zeros(1,length(lambda_c_MN_list));
for i=1:length(lambda_c_MN_list)
    for m=1:M
        output=cv_MN_node(X(1:(T_train-1),:,:),X_rounded(2:T_train,m,:),true,init_A,init_nu,eta,tol,iter);
        pred_err_arr(i)=pred_err_arr(i)+mean(cell2mat(output.pct_err));
    end
end
[~,I]=min(pred_err_arr);
lambda_c_MN=lambda_c_MN_list(I);
%fit MN model
Ah_MN=zeros(M,K,M,K);nu_h_MN=zeros(M,K);
for m=1:M
    [Ah_MN(m,:,:,:),nu_h_MN(m,:),~,~,~,~]=fit_MN_node(X(1:T,:,:),X_rounded(2:(T+1),:,:),lambda_c_MN*K*sqrt(log(M)/T),true,init_A, init_nu, eta,tol,iter);
end

%previous cross validation results for logistic-normal model can be
%directly used
lambda_c_LN=0.0061;alpha=0.3;
%fit MN model
Ah_LN=zeros(M,K,M,K);nu_h_LN=zeros(M,K);
for m=1:M
    [Ah_LN(m,:,:,:),nu_h_LN(m,:),~,~,~,~]=fit_LN_node(X(1:T,:,:),X(2:(T+1),:,:),lambda_c_LN*K*sqrt(log(M)/T),alpha,true,init_A, init_nu, eta,tol,iter);
end
   
%estimate a, sigma and Sigma
[ah,sigma_sq_hat]=est_a_sigma(X(1:T,:,:),X(2:(T+1),:,:),Ah_MN,nu_h_MN);
[logL_LN_est,logL_MN_est]=test_MN_LN(X(1:T,:,:),X(2:(T+1),:,:),Ah_MN,nu_h_MN,ah,sigma_sq_hat,Ah_LN,nu_h_LN);
L_diff=logL_LN_est-logL_MN_est;
save('test_statistic.mat','L_diff');

clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%cross validation for logistic-normal model with time-varying q
lambda_c_list=(0.0036:0.0005:0.0131)/K*sqrt(T_train/log(M));
alpha_list=[0.3 0.5 0.7];
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);eta=2;tol=0.0001;iter=500;
pred_err_arr=zeros(length(alpha_list),length(lambda_c_list));
for i=1:length(lambda_c_list)
    for j=1:length(alpha_list)
        output=cv_LN_Bern(X_train,lambda_c_list(i),alpha_list(j),true,init_A,init_nu,eta,tol,iter);
        pred_err_arr(j,i)=mean(cell2mat(output.pred_err));
    end
end
[J,I]=find(pred_err_arr==min(pred_err_arr(:)));
alpha=alpha_list(J);lambda_c_LN=lambda_c_list(I);

%training and prediction for the logistic-normal model with time-varying q
lambda_LN=lambda_c_LN*K*sqrt(log(M)/T_train);
[Ah_LN,nu_h_LN,~,~,~,~]=...
fit_LN_Bern(X_train,lambda_LN,alpha,true,init_A, init_nu, eta,tol,iter);
pred_err_LN=pred_err_LN_Bern(X_test,Ah_LN,nu_h_LN)/(T_test-1)
%prediction error: 0.144421374687766

%fit the whole data using the logistic-normal (time-varying q) model
lambda_LN=lambda_c_LN*K*sqrt(log(M)/T);
[Ah_LN,~,~,~,~,~]=fit_LN_Bern(X,lambda_LN,alpha,true,init_A, init_nu, eta,tol,iter);
save('fitted_LN.mat','Ah_LN','nu_h_LN');

%transfer the estimated parameters to variable importance networks
Infl_network_LN=var_importance(X(1:T,:,:),Ah_LN,nu_h_LN,1:M);
g_mat_LN=output_network(Infl_network_LN,0,true);
csvwrite('twitter_LN_infl_graph.csv',g_mat_LN);
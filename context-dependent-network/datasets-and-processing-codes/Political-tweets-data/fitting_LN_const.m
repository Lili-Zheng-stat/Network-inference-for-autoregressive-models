clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%fitting a constant process for LN model
nu_constant=zeros(M,K);
 for m=1:M
     ratio_inverse=T_train/sum(sum(X_train(:,m,:),3)>0);
     if ratio_inverse>1
        nu_constant(m,K)=-log(ratio_inverse-1);
     else 
         nu_constant(m,K)=50;
     end
     ind_temp=find(sum(X_train(:,m,:),3)~=0);
     nu_constant(m,1:(K-1))=reshape(mean(log(X_train(ind_temp,m,1:(K-1))./...
         X_train(ind_temp,m,K)),1),1,K-1);
 end
 pred_err_LN_const=pred_err_LN_Bern(X_test,zeros(M,K,M,K),nu_constant)
%predictione error: 0.158004112866873

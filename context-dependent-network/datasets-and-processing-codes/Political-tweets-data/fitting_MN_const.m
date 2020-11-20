clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;

%training and prediction for a constant multinomial process
p_hat_constant=reshape(mean(X_train_rounded,1),M,K);
[~,I_pred]=max([p_hat_constant 1-sum(p_hat_constant,2)],[],2)
I_truth=zeros(T_test-1,M);
for t=1:(T_test-1)
    for m=1:M
        if sum(X_test_rounded(t+1,m,:))==0
            I_truth(t,m)=K+1;
        else
            [~,I_truth(t,m)]=max(X_test_rounded(t+1,m,:));
        end
    end
end
pred_err_MN_const=mean(mean(double(I_truth~=repmat(reshape(I_pred,1,M),T_test-1,1))))
%0.305801948524066
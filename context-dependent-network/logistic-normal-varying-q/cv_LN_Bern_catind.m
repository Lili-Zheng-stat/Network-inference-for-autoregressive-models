function output=cv_LN_Bern_catind(X,lambda_constant,alpha,intercept,init_A,init_nu,eta,tol,iter)
%5-fold cross validataion for tuning parameter lambda_constant*sqrt(log(M)/T)),
%on data X, under the logistic-normal model with context-independent network
[T,M,~]=size(X);
T=T-1;
train_cv_sz=floor(0.8*(T+1));
lambda=lambda_constant*sqrt(log(M)/train_cv_sz);
output=struct;
for k=1:5
    init_test_sz=floor(0.05*(T+1)*(k-1));
    X_train_cv=X((init_test_sz+1):(train_cv_sz+init_test_sz),:,:);
    X_test_cv1=X(1:init_test_sz,:,:);
    X_test_cv2=X((train_cv_sz+init_test_sz+1):(T+1),:,:);
    
    [output.Ah{k},output.nu_h{k},output.Ah_BAR{k},output.nu_h_BAR{k},output.loss{k},output.grad_A{k},output.grad_nu{k},output.cvg{k}]=...
        fit_LN_Bern_catind(X_train_cv,lambda,intercept,init_A, init_nu, eta,tol,iter)
    
    %prediction
    if (init_test_sz>1)&&((T+1)>(train_cv_sz+init_test_sz+1))
        loss1=calc_loss_LN_Bern(output.Ah{k},output.nu_h{k},X_test_cv1,alpha);
        loss2=calc_loss_LN_Bern(output.Ah{k},output.nu_h{k},X_test_cv2,alpha);
        output.pred_loss{k}=(loss1*(init_test_sz-1)+loss2*...
            (T-train_cv_sz-init_test_sz))/(T-train_cv_sz-1);
        pred_err1=pred_err_LN_Bern(X_test_cv1,output.Ah{k},output.nu_h{k});
        pred_err2=pred_err_LN_Bern(X_test_cv2,output.Ah{k},output.nu_h{k});
        output.pred_err{k}=(pred_err1+pred_err2)/(T-train_cv_sz-1);
    elseif init_test_sz>1
        output.pred_loss{k}=calc_loss_LN_Bern(output.Ah{k},output.nu_h{k},X_test_cv1,alpha);
        output.pred_err{k}=pred_err_LN_Bern(X_test_cv1,output.Ah{k},output.nu_h{k})/(init_test_sz-1);
    else
        output.pred_loss{k}=calc_loss_LN_Bern(output.Ah{k},output.nu_h{k},X_test_cv2,alpha);
        output.pred_err{k}=pred_err_LN_Bern(X_test_cv2,output.Ah{k},output.nu_h{k})/(T-train_cv_sz-init_test_sz);
    end
end
end
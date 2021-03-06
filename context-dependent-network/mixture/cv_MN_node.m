function output=cv_MN_node(X_cov,X_resp,lambda_constant,intercept,init_A,init_nu,eta,tol,iter)
%5-fold cross validataion for tuning parameter lambda_constant*K*sqrt(log(M)/T)),
%on data X_cov (T*M*K) and X_resp (T*1*K).

[T,M,K]=size(X_cov);
train_cv_sz=floor(0.8*(T+1));
lambda=lambda_constant*K*sqrt(log(M)/(train_cv_sz-1));
output=struct;
for k=1:5
    init_test_sz=floor(0.05*(T+1)*(k-1));
    X_cov_train_cv=X_cov((init_test_sz+1):(train_cv_sz+init_test_sz-1),:,:);
    X_resp_train_cv=X_resp((init_test_sz+1):(train_cv_sz+init_test_sz-1),:,:);
    X_cov_test_cv1=X_cov(1:(init_test_sz-1),:,:);
    X_resp_test_cv1=X_resp(1:(init_test_sz-1),:,:);
    X_cov_test_cv2=X_cov((train_cv_sz+init_test_sz+1):T,:,:);
    X_resp_test_cv2=X_resp((train_cv_sz+init_test_sz+1):T,:,:);

    [output.Ah{k},output.nu_h{k},output.loss{k},output.grad_A{k},output.grad_nu{k},output.cvg{k}]=...
    fit_MN_node(X_cov_train_cv,X_resp_train_cv,lambda,intercept,init_A, init_nu, eta,tol,iter);

    %prediction
    if (init_test_sz>1)&&((T+1)>(train_cv_sz+init_test_sz+1))
        loss1=calc_loss_MN_node(output.Ah{k},output.nu_h{k},X_cov_test_cv1,X_resp_test_cv1);
        loss2=calc_loss_MN_node(output.Ah{k},output.nu_h{k},X_cov_test_cv2,X_resp_test_cv2);
        [pct_err1,prob_err1]=pred_err_MN_node(X_cov_test_cv1,X_resp_test_cv1,output.Ah{k},output.nu_h{k});
        [pct_err2,prob_err2]=pred_err_MN_node(X_cov_test_cv2,X_resp_test_cv2,output.Ah{k},output.nu_h{k});
        output.pred_loss{k}=(loss1*(init_test_sz-1)+loss2*...
            (T-train_cv_sz-init_test_sz))/(T-train_cv_sz-1);
        output.pct_err{k}=(pct_err1+pct_err2)/(T-train_cv_sz-1);
        output.prob_err{k}=(prob_err1+prob_err2)/(T-train_cv_sz-1);
    elseif init_test_sz>1
        output.pred_loss{k}=calc_loss_MN_node(output.Ah{k},output.nu_h{k},X_cov_test_cv1,X_resp_test_cv1);
        [pct_err,prob_err]=pred_err_MN_node(X_cov_test_cv1,X_resp_test_cv1,output.Ah{k},output.nu_h{k});
        output.pct_err{k}=pct_err/(init_test_sz-1);
        output.prob_err{k}=prob_err/(init_test_sz-1);
    else
        output.pred_loss{k}=calc_loss_MN_node(output.Ah{k},output.nu_h{k},X_cov_test_cv2,X_resp_test_cv2);
        [pct_err,prob_err]=pred_err_MN_node(X_cov_test_cv2,X_resp_test_cv2,output.Ah{k},output.nu_h{k});
        output.pct_err{k}=pct_err/(T-train_cv_sz-init_test_sz);
        output.prob_err{k}=prob_err/(T-train_cv_sz-init_test_sz);
    end

end

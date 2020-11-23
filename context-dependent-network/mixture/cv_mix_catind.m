function output=cv_mix_catind(X,LN_list,lambda_constant,intercept,init_A,init_nu,eta,tol,iter)
%5-fold cross validataion for tuning parameter lambda_constant*sqrt(log(M)/T)),
%on data X, under the mixture model with context-independent network
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
    
    [output.Ah_BAR{k},output.nu_h_BAR{k},output.loss{k},output.grad_A{k},output.grad_nu{k},output.cvg{k}]=...
        fit_BAR(sum(X_train_cv,3),lambda,intercept,init_A, init_nu, eta,tol,iter)
    
    %prediction
    Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
    pred_err=zeros(M,T-train_cv_sz-1);
    for m=1:M
        ind_temp=find(sum(X_train_cv(:,m,:),3)~=0);
        if sum(LN_list==m)>0
            Ah(m,K,:,:)=repmat(reshape(output.Ah_BAR{k}(m,:),1,1,M,1),1,1,1,K);
            nu_h(m,K)=output.nu_h_BAR{k}(m);
            nu_h(m,1:(K-1))=reshape(mean(log(X_train_cv(ind_temp,m,1:(K-1))./...
            X_train_cv(ind_temp,m,K)),1),1,K-1);
            if (init_test_sz>1)&&(T_train>train_cv_sz+init_test_sz+1)
                pred_err(m,1:(init_test_sz-1))=pred_err_LN_node(X_test_cv1(1:(init_test_sz-1),:,:),X_test_cv1(2:init_test_sz,m,:),Ah(m,:,:,:),nu_h(m,:));
                pred_err(m,init_test_sz:(T_train-train_cv_sz-2))=pred_err_LN_node(X_test_cv2(1:(T_train-train_cv_sz-init_test_sz-1),:,:),X_test_cv2(2:(T_train-train_cv_sz-init_test_sz),m,:),Ah(m,:,:,:),nu_h(m,:));
            elseif init_test_sz>1
                pred_err(m,:)=pred_err_LN_node(X_test_cv1(1:(init_test_sz-1),:,:),X_test_cv1(2:init_test_sz,m,:),Ah(m,:,:,:),nu_h(m,:));
            else
                pred_err(m,:)=pred_err_LN_node(X_test_cv2(1:(T_train-train_cv_sz-init_test_sz-1),:,:),X_test_cv2(2:(T_train-train_cv_sz-init_test_sz),m,:),Ah(m,:,:,:),nu_h(m,:));
            end
        else
            Ah(m,:,:,:)=repmat(reshape(output.Ah_BAR(m,:),1,1,M,1),1,K,1,K);
            p_hat=mean(X_train_cv(ind_temp,m,:),1);
            nu_h(m,:)=repmat(output.nu_h_BAR(m),1,K)+reshape(log(p_hat),1,K);
            if (init_test_sz>1)&&(T_train>train_cv_sz+init_test_sz+1)
                [pred_err(m,1:(init_test_sz-1)),~]=pred_err_MN_node(X_test_cv1(1:(init_test_sz-1),:,:),X_test_cv1(2:init_test_sz,m,:),Ah(m,:,:,:),nu_h(m,:));
                [pred_err(m,init_test_sz:(T_train-train_cv_sz-2)),~]=pred_err_MN_node(X_test_cv2(1:(T_train-train_cv_sz-init_test_sz-1),:,:),X_test_cv2(2:(T_train-train_cv_sz-init_test_sz),m,:),Ah(m,:,:,:),nu_h(m,:));
            elseif init_test_sz>1
                [pred_err(m,:),~]=pred_err_MN_node(X_test_cv1(1:(init_test_sz-1),:,:),X_test_cv1(2:init_test_sz,m,:),Ah(m,:,:,:),nu_h(m,:));
            else
                [pred_err(m,:),~]=pred_err_MN_node(X_test_cv2(1:(T_train-train_cv_sz-init_test_sz-1),:,:),X_test_cv2(2:(T_train-train_cv_sz-init_test_sz),m,:),Ah(m,:,:,:),nu_h(m,:));
            end
        end
    end
    output.pred_err{k}=pred_err;output.Ah{k}=Ah;output.nu_h{k}=nu_h;
end
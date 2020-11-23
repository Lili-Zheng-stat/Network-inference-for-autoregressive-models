function [logLR,Ah_MN,nu_h_MN,Ah_LN,nu_h_LN,ah,sigma_MN_sq_hat]=testing(X,nlambda)
%Test the type of each node
[T,M,K]=size(X);
T=T-1;
X_rounded = zeros(T+1,M,K);
for i=1:(T+1)
    for j=1:M
        if max(X(i,j,:))>0
           [~,I]=sort(X(i,j,:),'descend');
           X_rounded(i,j,I(1))=1;
        end
    end
end

%fitting MN and LN models by regressing rounded and original future data
%upon original past data
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
eta=2;tol=0.0001;iter=1000;
%cross validation for multinomial model
lambda_c_max=0.01;stop=false;
while ~stop
    lambda=lambda_c_max*K*sqrt(log(M)/T);
    Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
    for m=1:M
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_MN_node(X(1:T,:,:),X_rounded(2:(T+1),m,:),lambda,true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
    end
    if sum(Ah(:).^2)>0
        lambda_c_max=lambda_c_max*2;
        init_A=Ah;init_nu=nu_h;
    else
        stop=true;
    end
end
lambda_c_min=lambda_c_max*0.001;
lambda_c_MN_list=exp(linspace(log(lambda_c_min),log(lambda_c_max),nlambda));
pred_err_arr=zeros(1,length(lambda_c_MN_list));
for i=1:length(lambda_c_MN_list)
    Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
    for m=1:M
        output=cv_MN_node(X(1:T,:,:),X_rounded(2:(T+1),m,:),lambda_c_MN_list(i),true,init_A(m,:,:,:),init_nu(m,:),eta,tol,iter);
        pred_err_arr(i)=pred_err_arr(i)+mean(cell2mat(output.pct_err));
        for j=1:5
            Ah(m,:,:,:)=Ah(m,:,:,:)+output.Ah{j};
            nu_h(m,:,:,:)=nu_h(m,:,:,:)+output.nu_h{j};
        end
    end
    init_A=Ah/5;init_nu=nu_h/5;
end
[~,I]=min(pred_err_arr);
lambda_c_MN=lambda_c_MN_list(I);
%fit MN model
Ah_MN=zeros(M,K,M,K);nu_h_MN=zeros(M,K);
for m=1:M
    [Ah_MN(m,:,:,:),nu_h_MN(m,:),~,~,~,~]=fit_MN_node(X(1:T,:,:),X_rounded(2:(T+1),m,:),lambda_c_MN*K*sqrt(log(M)/T),true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
end

%cross validation for logistic-normal model
lambda_c_max=0.01;stop=false;alpha_list=[0.3 0.5 0.7];
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
for j=1:length(alpha_list)
    while ~stop
        lambda=lambda_c_max*K*sqrt(log(M)/T);
        Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
        for m=1:M
            [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_LN_node(X(1:T,:,:),X(2:(T+1),m,:),lambda,alpha_list(j),true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
        end
        if sum(Ah(:).^2)>0
            lambda_c_max=lambda_c_max*2;
            init_A=Ah;init_nu=nu_h;
        else
            stop=true;
        end
    end
    stop=false;
end
lambda_c_min=lambda_c_max*0.001;
lambda_c_LN_list=exp(linspace(log(lambda_c_min),log(lambda_c_max),nlambda));
pred_err_arr=zeros(length(alpha_list),length(lambda_c_LN_list));
for i=1:length(alpha_list)
    for j=1:length(lambda_c_LN_list)
        Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
        for m=1:M
            output=cv_LN_node(X(1:T,:,:),X(2:(T+1),m,:),lambda_c_LN_list(j),alpha_list(i),true,init_A(m,:,:,:),init_nu(m,:),eta,tol,iter);
            pred_err_arr(i,j)=pred_err_arr(i,j)+mean(cell2mat(output.pred_err));
            for k=1:5
                Ah(m,:,:,:)=Ah(m,:,:,:)+output.Ah{k};
                nu_h(m,:,:,:)=nu_h(m,:,:,:)+output.nu_h{k};
            end
        end
        init_A=Ah/5;init_nu=nu_h/5;
    end
end
[I,J]=find(pred_err_arr==min(pred_err_arr(:)));
alpha=alpha_list(I);lambda_c_LN=lambda_c_LN_list(J);
%fit LN model
Ah_LN=zeros(M,K,M,K);nu_h_LN=zeros(M,K);
for m=1:M
    [Ah_LN(m,:,:,:),nu_h_LN(m,:),~,~,~,~]=fit_LN_node(X(1:T,:,:),X(2:(T+1),m,:),lambda_c_LN*K*sqrt(log(M)/T),alpha,true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
end
   
%estimate a, sigma and Sigma
[ah,sigma_MN_sq_hat]=est_a_sigma(X(1:T,:,:),X(2:(T+1),:,:),Ah_MN,nu_h_MN);
[logL_LN_est,logL_MN_est]=test_MN_LN(X(1:T,:,:),X(2:(T+1),:,:),Ah_MN,nu_h_MN,ah,sigma_MN_sq_hat,Ah_LN,nu_h_LN);
logLR=logL_LN_est-logL_MN_est;
end

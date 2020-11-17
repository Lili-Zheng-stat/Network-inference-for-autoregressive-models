clc
clear all

load('X.mat');
[T,M,K]=size(X);
T=T-1;
X_mat=reshape(X,(T+1)*M,K);

%Generate histogram of right leaning scores of tweets
histogram(X_mat(sum(X_mat,2)>0,2),50,'Normalization','count')
xlabel('Right-leaning scores','Fontsize',18);ylabel('frequency','Fontsize',18)
saveas(gca,'tweets_right-leaning_scores.png')

%Generate rounded data: used for multinomial modeling
X_rounded = zeros(T+1,M,K);
for i=1:(T+1)
    for j=1:M
        if max(X(i,j,:))>0
           [~,I]=sort(X(i,j,:),'descend');
           X_rounded(i,j,I(1))=1;
        end
    end
end

X_train=X(1:floor(0.7*(T+1)),:,:);T_train=floor(0.7*(T+1));
X_train_rounded=X_rounded(1:floor(0.7*(T+1)),:,:);
X_test=X((floor(0.7*(T+1))+1):(T+1),:,:);
X_test_rounded=X_rounded((floor(0.7*(T+1))+1):(T+1),:,:);[T_test,~,~]=size(X_test);

%cross validation for the multinomial model
lambda_c_list=(0.005:0.0001:0.0085)/K*sqrt(T/log(M));
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);eta=2;tol=0.0001;iter=500;
pred_err_arr=zeros(1,length(lambda_c_list));
for i=1:length(lambda_c_list)
    output=cv_MN(X_train,lambda_c_list,true,init_A,init_nu,eta,tol,iter);
    pred_err_arr(i)=mean(output.pred_err);
end
[~,ind]=min(pred_err_arr);
lambda_c_MN=lambda_c_list(ind);

%training and prediction for the multinomial model
lambda_MN=lambda_c_MN*K*sqrt(log(M)/T);
output=struct;
[output.Ah_MN,output.nu_h_MN,output.loss,output.grad_A,output.grad_nu,output.cvg]=...
fit_MN(X_train,lambda,true,init_A, init_nu, eta,tol,iter);
pred_err_MN=pred_err_MN(X_test_rounded,output.Ah_MN,output.nu_h_MN)/(T_test-1)
%0.251999418351025

%training and prediction for the multinomial model with
%category-independent network
lambda=0.001;
init_A=zeros(M,M);init_nu=zeros(M,1);
eta=2;tol=0.0001;iter=2000;
output=struct;
[output.Ah_BAR,output.nu_h_BAR,output.loss,output.grad_A,output.grad_nu,output.cvg]=...
    fit_BAR(sum(X_train_rounded,3),lambda,true,init_A, init_nu, eta,tol,iter);
Ah_BG=repmat(reshape(output.Ah_BAR,M,1,M,1),1,K,1,K);
p_hat=zeros(M,K);
 for m=1:M
     ind_temp=find(sum(X_train_rounded(:,m,:),3)~=0);
     p_hat(m,:)=mean(X_train_rounded(ind_temp,m,:),1);
 end
nu_h_BG=repmat(output.nu_h_BAR,1,K)+log(p_hat);
pred_err_catind=pred_err_multinomial(X_test_rounded,Ah_BG,nu_h_BG)/(T_test-1)
%0.255198487712665

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
mean(mean(double(I_truth~=repmat(reshape(I_pred,1,M),T_test-1,1))))
%0.305801948524066

%cross validation for catind_LN
load('bert_Q.mat');
[T,M,K]=size(Q);
T=T-1;
X_train=Q(1:floor(0.7*(T+1)),:,:);T_train=floor(0.7*(T+1));
train_cv_sz=floor(0.8*T_train);
lambda_list=0.001:0.001:0.009;
cv_catind_LN_err=zeros(9,5);
for i=1:9
    for j=1:5
        init_test_sz=floor(0.05*T_train*(j-1));
        Q_train_cv=X_train((init_test_sz+1):(train_cv_sz+init_test_sz),:,:);
        Q_test_cv1=X_train(1:init_test_sz,:,:);
        Q_test_cv2=X_train((train_cv_sz+init_test_sz+1):T_train,:,:);
        filename=sprintf('fitting/cv_catind1_%d_%d.mat',i,j);
        load(filename);
        Ah_BG=zeros(M,K,M,K);nu_h_BG=zeros(M,K);
        Ah_BG(:,K,:,:)=repmat(reshape(output.Ah_BAR,M,1,M,1),1,1,1,K);
        nu_h_BG(:,K)=reshape(output.nu_h_BAR,M,1);
        for m=1:M
            ind_temp=find(sum(Q_train_cv(:,m,:),3)~=0);
            nu_h_BG(m,1:(K-1))=reshape(mean(log(Q_train_cv(ind_temp,m,1:(K-1))./...
            Q_train_cv(ind_temp,m,K)),1),1,K-1);
        end
        %prediction
        if (init_test_sz>1)&&(T_train>train_cv_sz+init_test_sz+1)
           [~, ~, ~,~,total_err1]=pred_error_sum(Q_test_cv1,Ah_BG,nu_h_BG);
           [~,~,~,~,total_err2]=pred_error_sum(Q_test_cv2,Ah_BG,nu_h_BG);
           output.pred_err_LN=(total_err1+total_err2)/(T_train-train_cv_sz-2);
        elseif init_test_sz>1
           [~,~,~,output.pred_err_LN]=pred_error(Q_test_cv1,Ah_BG,nu_h_BG);
        else
           [~,~,~,output.pred_err_LN]=pred_error(Q_test_cv2,Ah_BG,nu_h_BG);
        end
        save(filename,'output');
        cv_catind_LN_err(i,j)=output.pred_err_LN;
    end
end
mean(cv_catind_LN_err,2)
lambda_list(4)%lambda=0.004

%prediction performance of catind_LN
X_test=Q((floor(0.7*(T+1))+1):(T+1),:,:);[T_test,~,~]=size(X_test);
lambda=0.004;
init_A=zeros(M,M);init_nu=zeros(M,1);
eta=2;tol=0.0001;iter=2000;
output=struct;
[output.Ah_BAR,output.nu_h_BAR,output.loss,output.grad_A,output.grad_nu,output.cvg]=...
    fit_BAR(sum(X_train,3),lambda,true,init_A, init_nu, eta,tol,iter);
Ah_BG=zeros(M,K,M,K);nu_h_BG=zeros(M,K);
Ah_BG(:,K,:,:)=repmat(reshape(output.Ah_BAR,M,1,M,1),1,1,1,K);
nu_h_BG(:,K)=reshape(output.nu_h_BAR,M,1);
for m=1:M
    ind_temp=find(sum(X_train(:,m,:),3)~=0);
    nu_h_BG(m,1:(K-1))=reshape(mean(log(X_train(ind_temp,m,1:(K-1))./...
         X_train(ind_temp,m,K)),1),1,K-1);
end
[~, ~, ~,output.total_err]=pred_error(X_test,Ah_BG,nu_h_BG);
%prediction error: 0.143732477362186


%cv for GSM
cv_GSM_err=zeros(3,30,5);
for i=1:3
    for j=1:30
        for k=1:5
            load(sprintf('fitting_cv_GSM/cv_GSM1_%d_%d_%d.mat',i,j,k));
            cv_GSM_err(i,j,k)=output.total_err;
        end
    end
end
mean(cv_GSM_err,3)
lambda_list=(0.0025:0.0001:0.0045);lambda_list(12)
%lambda=0.0036;

%after corrected fit_R_alpha
%cv for GSM
cv_GSM_err=zeros(3,20,5);
for i=1:3
    for j=1:20
        for k=1:5
            load(sprintf('fitting_new/cv_GSM1_%d_%d_%d.mat',i,j,k));
            cv_GSM_err(i,j,k)=output.total_err;
        end
    end
end
mean(cv_GSM_err,3)
lambda_list=(0.0036:0.0005:0.0131);lambda_list(6)
cv_GSM_err=zeros(9,5);
    for j=1:9
        for k=1:5
            load(sprintf('fitting_new/cv_GSM1_%d_%d_%d.mat',1,j+20,k));
            cv_GSM_err(j,k)=output.total_err;
        end
    end
mean(cv_GSM_err,2)
%the best is 0.0061

lambda=0.0061;alpha=0.3;
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
eta=2;tol=0.0001;iter=500;
output=struct;
[output.Ah_joint,output.nu_h_joint,output.loss,output.grad_A,output.grad_nu,output.cvg]=...
fit_R_alpha(X_train,lambda,alpha,1,true,init_A, init_nu, eta,tol,iter);
[~,~,~,output.total_err]=pred_error(X_test,output.Ah_joint,output.nu_h_joint);
%prediction error: 0.144421374687766

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
 [~, ~, ~,total_err_constant]=pred_error(X_test,zeros(M,K,M,K),nu_constant)
%predictione error: 0.158004112866873



%fit the whole data using mult and GSM
[T,M,K]=size(Q);T=T-1;T_train=floor(0.7*(T+1));
lambda_mult=0.0054*sqrt(T_train/T);
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
eta=2;tol=0.0001;iter=500;
[Ah_mult,~,~,~,~,cvg]=fit_mult(Q,lambda_mult,true,init_A, init_nu, eta,tol,iter);
thrs=0;
relative_network_mult=output_network_relative(Ah_mult,thrs,string('multinomial'),true);
csvwrite('mult_graph_relative.csv',relative_network_mult);
network_mult=output_network_across_topics(Ah_mult,thrs,string('multinomial'),1,true);
csvwrite('mult_graph.csv',network_mult);

%lambda_GSM=0.0036*sqrt(T_train/T);
%after corrected fit_R_alpha
lambda_GSM=0.0061*sqrt(T_train/T);
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
eta=2;tol=0.0001;iter=500;
[Ah_GSM,~,~,~,~,cvg]=fit_R_alpha(Q,lambda_GSM,0.3,1,true,init_A, init_nu, eta,tol,iter);
relative_network_GSM=output_network_relative(Ah_GSM,thrs,string('mixed'),true);
%csvwrite('GSM_graph_relative.csv',relative_network_GSM);
%after corrected fit_R_alpha
csvwrite('GSM_graph_relative_new.csv',relative_network_GSM);

network_GSM=output_network_across_topics(Ah_GSM,thrs,string('mixed'),1,true);
csvwrite('GSM_graph.csv',network_GSM);


%reproduce previous results???
[Ah_joint,nu_h_joint,~,~,~,cvg]=fit_R_alpha(Q(1:floor(0.8*(T+1)),:,:),0.004,0.2,1,true,init_A, init_nu, eta,tol,iter);
%calc_loss_GSM(Ah_joint,nu_h_joint,Q((floor(0.8*(T+1))+1):(T+1),:,:),0)
%calc_loss_GSM(Ah_joint,nu_h_joint,Q((floor(0.8*(T+1))+1):(T+1),:,:),1)
%[prob_err, pct_err, weight_err,logrt_err]=pred_error(Q((floor(0.8*(T+1))+1):(T+1),:,:),...
%Ah_joint,nu_h_joint)
csvwrite('bert_reproducing_gsm.csv',...
output_network(Ah_joint,thrs,string('mixed'),1,true));
lambda_mult=0.008/sqrt(T/1000);
[Ah_mult,nu_h_mult,loss,grad_A,grad_nu,cvg]=fit_mult(Q(1:floor(0.8*(T+1)),:,:),lambda_mult, ...
true,init_A, init_nu, eta,tol,iter);
csvwrite('bert_reproducing_mult.csv',...
output_network(Ah_mult,thrs,string('multinomial'),1,true));


%prediction?
[prob_err, pct_err, weight_err,logrt_err]=pred_error(X_test,Ah_joint,nu_h_joint)
neg_log_likelihood_dis=calc_loss_GSM(Ah_joint,nu_h_joint,X_test,0)
neg_log_likelihood_cts=calc_loss_GSM(Ah_joint,nu_h_joint,X_test,1)

[prob_err0, pct_err0, weight_err0]=pred_error(X_test,zeros(M,K,M,K),nu_h_joint)
neg_log_likelihood0_dis=calc_loss_GSM(zeros(M,K,M,K),nu_h_joint,X_test,0)
neg_log_likelihood0_cts=calc_loss_GSM(zeros(M,K,M,K),nu_h_joint,X_test,1)

 Bh_joint=zeros(M,K,M,K);
 Bh_joint(:,K,:,:)=(Ah_joint(:,K,:,:)-sum(Ah_joint(:,1:(K-1),:,:),2))/K;
 Bh_joint(:,1:(K-1),:,:)=Ah_joint(:,1:(K-1),:,:)+Ah_joint(:,K,:,:);
 %Ah1=Ah_joint;
 %Ah2=Ah_joint;

link_joint=zeros(M,M);
for m=1:M
    for n=1:M
        link_joint(m,n)=norm(reshape(Ah_joint(m,:,n,:),K,K),'fro');
    end
end
users=users([1:4 5:7 9:17 19:25]);
labels=labels([1:4 5:7 9:17 19:25]);
%link(find(labels==string('dem')),find(labels==string('dem')))
%link(find(labels==string('dem')),find(labels==string('rep')))
%link(find(labels==string('rep')),find(labels==string('rep')))
%link(find(labels==string('rep')),find(labels==string('dem')))

%fit two BARs
%Y_train=(Q_train~=0);
%init_A=zeros(M,M);init_nu=zeros(M,1);lambda_BAR=0.01/sqrt(T/1000);
%[Ah1_BAR,nu_h1_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(reshape(Y_train(:,:,1),floor(0.8*(T+1)),M),lambda_BAR,...
%true,init_A, init_nu, eta,tol,iter);
%[Ah2_BAR,nu_h2_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(reshape(Y_train(:,:,2),floor(0.8*(T+1)),M),lambda_BAR,...
%true,init_A, init_nu, eta,tol,iter);

%fit multinomial
X_rounded = zeros(T+1,M,K);
for i=1:(T+1)
    for j=1:M
        if max(Q(i,j,:))>0
           [~,I]=sort(Q(i,j,:),'descend');
           X_rounded(i,j,I(1))=1;
        end
    end
end
X_test_rounded=X_rounded((floor(0.8*(T+1))+1):(T+1),:,:);
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);lambda_mult=0.008/sqrt(T/1000);
[Ah_mult,nu_h_mult,loss,grad_A,grad_nu,cvg]=fit_mult(X_train,lambda_mult, ...
true,init_A, init_nu, eta,tol,iter);

calc_loss_mult(Ah_mult,nu_h_mult,X_test_rounded)
pred_err_mult=pred_err_multinomial(X_test_rounded,Ah_mult,nu_h_mult)
[pct_err_mult0, weight_err_mult0,logrt_err_mult0]=...
    pred_err_multinomial(zeros(M,K,M,K),nu_h_mult,X_test,X_test_rounded)
Bh_mult=zeros(M,K,M,K);
Bh_mult(:,K,:,:)=(Ah_mult(:,K,:,:)-sum(Ah_mult(:,1:(K-1),:,:),2))/K;
Bh_mult(:,1:(K-1),:,:)=Ah_mult(:,1:(K-1),:,:)+Ah_mult(:,K,:,:);
link_mult=zeros(M,M);
for m=1:M
    for n=1:M
        link_mult(m,n)=norm(reshape(Ah_mult(m,:,n,:),K,K),'fro');
    end
end

prob_constant=reshape(mean(X_rounded(1:floor(0.8*(T+1)),:,:),1),M,K);
nu_constant=log(prob_constant./(1-sum(prob_constant,2)));
calc_loss_mult(zeros(M,K,M,K),nu_constant,X_test_rounded)
[pct_err01_constant,pct_err_constant, weight_err_constant,logrt_err_constant]=...
    pred_err_multinomial(zeros(M,K,M,K),nu_constant,X_test,X_test_rounded)


%fit a model with rounded data as input and output weight data, to show the
%loss of rounding
lambda=0.005
[Ah_rounding,nu_h_rounding,loss,grad_A,grad_nu,cvg]=fit_R_alpha_rounded(X_train,lambda, ...
alpha,1,true,init_A, init_nu, eta,tol,iter);
[prob_err_rounding, pct_err_rounding, weight_err_rounding,logrt_err_rounding]=...
    pred_error_rounded(X_test,Ah_rounding,nu_h_rounding)
neg_log_likelihood_dis=calc_loss_GSM(Ah_joint,nu_h_joint,X_test,0)
neg_log_likelihood_cts=calc_loss_GSM(Ah_joint,nu_h_joint,X_test,1)


%fit BAR
init_A=zeros(M,M);init_nu=zeros(M,1);lambda_BAR=0.005/sqrt(T/1000);
[Ah_BAR,nu_h_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(sum(X_train,3),lambda_BAR, ...
true,init_A, init_nu, eta,tol,iter);
calc_loss_BAR(Ah_BAR,nu_h_BAR,sum(X_test,3))
pct_err_BAR=pred_err_BAR(sum(X_test,3),Ah_BAR,nu_h_BAR)

[Ah1_BAR,nu_h1_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(double(X_train(:,:,1)~=0),lambda_BAR, ...
true,init_A, init_nu, eta,tol,iter);
[Ah2_BAR,nu_h2_BAR,loss,grad_A,grad_nu,cvg]=fit_BAR(double(X_train(:,:,2)~=0),lambda_BAR, ...
true,init_A, init_nu, eta,tol,iter);
%compare prediction between GSM_joint and multinomial
pct_err_mult=pct_err_multinomial(Ah_mult,nu_h_mult,X_test)
pct_err_GSM=pct_err_GSM_rounding(Ah_joint,nu_h_joint,X_test)
pct_err_BAR=pct_err_separate_BAR(Ah1_BAR,nu_h1_BAR,Ah2_BAR,nu_h2_BAR,X_test)





thrs=0;
csvwrite('bert2.csv',...
output_network(Ah_joint,thrs,string('mixed'),1,true));
network=output_network(Ah_joint,thrs,string('mixed'),2,true);
thrs_plot=0;
sum(labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('rep'))
sum(labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==1&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('rep'))

thrs_plot=0.2;
sum(labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('rep'))
sum(labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==1&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('rep'))

thrs_plot=0.2;
sum(labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('rep'))
sum(labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==2&network(:,5)==1&network(:,3)>thrs_plot,2))...
    ==string('rep'))

thrs_plot=0.2;
sum(labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('dem')&labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('rep'))
sum(labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('dem'))
sum(labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,1))...
    ==string('rep')&labels(network(network(:,4)==2&network(:,5)==2&network(:,3)>thrs_plot,2))...
    ==string('rep'))




csvwrite('bert1_mult.csv',...
output_network(Ah_mult,thrs,string('multinomial'),1,true));

Ah_mult_relative=zeros(M,K,M,K);
Ah_mult_relative(:,1:(K-1),:,:)=Ah_mult(:,1:(K-1),:,:)-Ah_mult(:,K,:,:);
Ah_mult_relative(:,K,:,:)=sum(Ah_mult(:,1:(K-1),:,:),2);
csvwrite('bert1_mult_relative.csv',...
output_network(Ah_mult_relative,thrs,string('multinomial'),1,true));

csvwrite('bert1_rounding.csv',...
output_network(Ah_rounding,thrs,string('mixed'),true));

%check the influence of Senate Democrats on John Kasich
source=find(users==string('Senate Democrats'));
target=find(users==string('John Kasich'));
t_ind=find(sum(X_train(2:floor(0.8*(T+1)),target,:),3)>0);
a=2:floor(0.8*(T+1));
figure(3)
plot(X_train(a(t_ind)-1,source,1),...
    log(X_train(a(t_ind),target,1)./X_train(a(t_ind),target,2)),'.');
hold on;
plot([0 1],[0 0],'LineStyle','--','Linewidth',2);
plot([0.5 0.5],[-6 4],'LineStyle','--','Linewidth',2);
hold off

plot(X_train(a(t_ind)-1,source,1),...
    log(X_train(a(t_ind),target,1)./X_train(a(t_ind),target,2))>0,'.');
figure(1)
hist(X_train(a(t_ind(X_train(a(t_ind),target,1)>0.5))-1,source,1))
figure(2)
hist(X_train(a(t_ind(X_train(a(t_ind),target,1)<0.5))-1,source,1))



# Estimating context-dependent network in autoregressive point process models
# Introduction


This is a tutorial for how to use the MATLAB functions in this folder to conduct estimation for context-dependent networks from time series data, based on autoregressive point process modeling. The methodologies are proposed in the paper [Context-dependent self-exciting point processes: models, methods, and risk bounds in high dimensions](https://arxiv.org/abs/2003.07429), including a multinomial approach which assumes a multinomial autoregressive model, a logistic-normal approach that assumes a logistic-normal autoregressive model, and a mixture approach assuming some nodes in the network follow the multinomial model while others follw the logistic-normal model. 




In this tutorial, we illustrate how to apply the multinomial, logistic-normal and mixture approaches through some examples.


# Multinomial method


Assume that the time series data <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;X^t&space;\in&space;R^{M\times&space;K}&space;\rbrace_{t=0}^T"/> follow the multinomial autoregressive model defined in the paper, with network parameter <img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{MN}&space;\in&space;R^{M\times&space;K\times&space;M\times&space;K}"/> and offset parameter <img src="https://latex.codecogs.com/gif.latex?\inline&space;\nu^{MN}&space;\in&space;R^{M\times&space;K}"/>, then we can estimate<img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{MN}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\nu^{MN}"/> by applying the functions in the folder **multinomial.**




***Generate data:***




First we consider an example with <img src="https://latex.codecogs.com/gif.latex?\inline&space;M=10"/> nodes, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;K=2"/> categories, each node is influenced by <img src="https://latex.codecogs.com/gif.latex?\inline&space;\rho&space;=1"/> randomly chosen node. Each edge weight follows uniform distribution <img src="https://latex.codecogs.com/gif.latex?\inline&space;U(-2,2)"/>, and the offset parameter is set to ensure the event probability to be `prob=0.8` when no past influence exists.



```matlab:Code
addpath('multinomial')
M=10;K=2;rho=1;
rng(2578);
A=zeros(M,K,M,K);
for i=1:M
    connect=randsample(M,rho);
    for k=1:size(connect)
     A(i,:,connect(k),:)=(rand(K,K)-0.5)*4;
    end
end
prob=0.8;
nu=ones(M,K)*log(prob/K/(1-prob));
```



Then we generate multinomial time series data <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;X^t&space;\rbrace_{t=0}^T"/> with sample size <img src="https://latex.codecogs.com/gif.latex?\inline&space;T=1000"/>.



```matlab:Code
T=1000;
X=data_gen_MN(M,K,T,A,nu,prob);
```



***Cross validation:***




Given data<img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;X^t&space;\rbrace_{t=0}^T"/>, we can run 5-fold cross validation using function `cv_MN.m`.` `First we need to specify a list of tuning parameters for us to choose from. We can let the largest one <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\max&space;}"/> to be the smallest <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda"/> leading to zero estimate for <img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{MN}"/>, and then set the smallest one <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\min&space;}"/> to be <img src="https://latex.codecogs.com/gif.latex?\inline&space;0.001\times&space;\lambda_{\max&space;}"/>. We then choose the list of tuning parameters to be equally spaced between <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\min&space;}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\max&space;}"/> under the log-scale.



```matlab:Code
init_nu=zeros(M,K);
init_A=zeros(M,K,M,K);
eta=2;tol=0.0001;iter=500;
```



Choose <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\max&space;}"/>:



```matlab:Code
lambda_max=0.01;stop=false;
while ~stop
    lambda=lambda_max*K*sqrt(log(M)/T);
    [Ah,~,~,~,~,~]=fit_MN(X,lambda,true,init_A, init_nu, eta,tol,iter);
    if sum(Ah(:).^2)>0
        lambda_max=lambda_max*2;
    else
        stop=true;
    end
end
```



Choose <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\min&space;}"/>:



```matlab:Code
lambda_min=lambda_max*0.001;
```



Generate evenly spaced 20 tuning parameters under the log-scale:



```matlab:Code
lambda_c_list=exp(linspace(log(lambda_min),log(lambda_max),20));
```



Given the list of tuning parameters, we run 5-fold cross validation:



```matlab:Code
pred_err=zeros(1,length(lambda_c_list));pred_loss=zeros(1,length(lambda_c_list));
for i=1:length(lambda_c_list)
    output=cv_MN(X,lambda_c_list(i),true,init_A,init_nu,eta,tol,iter);
    pred_loss(i)=mean(cell2mat(output.pred_loss));
    pred_err(i)=mean(cell2mat(output.pred_err));
end  
```



`pred_loss` and `pred_err `contain the average log-likelihood loss and prediction errors evaluated at the test sets in 5-fold cross validation, and either one can be used as the criterion for choosing lambda.



```matlab:Code
[~,I]=min(pred_loss);
```



***Estimate the parameters:***



```matlab:Code
lambda=lambda_c_list(I)*K*sqrt(log(M)/T);
[Ah,nu_h,~,~,~,~]=fit_MN(X,lambda,true,init_A, init_nu, eta,tol,iter);
```



Estimation error:



```matlab:Code
sqrt(sum((Ah(:)-A(:)).^2))
```


```text:Output
ans = 2.1755
```


```matlab:Code
sqrt(sum((nu_h(:)-nu(:)).^2))
```


```text:Output
ans = 1.0627
```

# Logistic-normal method


Assume that the time series data <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;X^t&space;\in&space;R^{M\times&space;K}&space;\rbrace_{t=0}^T"/> follow the logistic-normal autoregressive model with event probability varying over time, where the network parameters are<img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{LN}&space;\in&space;R^{M\times&space;(K-1)\times&space;M\times&space;K}"/>, <img src="https://latex.codecogs.com/gif.latex?\inline&space;B^{Bern}&space;\in&space;R^{M\times&space;1\times&space;M\times&space;K}"/> and offset parameters are <img src="https://latex.codecogs.com/gif.latex?\inline&space;\nu^{LN}&space;\in&space;R^{M\times&space;(K-1)}"/>, <img src="https://latex.codecogs.com/gif.latex?\inline&space;\eta^{Bern}&space;\in&space;R^{M\times&space;1}"/>. We can estimate these parameters by applying the functions in the folder **logistic-normal-varying-q. **For simplicity of the codes, we will use only two parameters `A `and `nu`, where `A` is concatenated by <img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{LN}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;B^{Bern}"/> in the second dimension, `nu` is concatenated by <img src="https://latex.codecogs.com/gif.latex?\inline&space;\nu^{LN}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\eta^{Bern}"/>in the second dimension. When assuming the event probability for each node is a constant (<img src="https://latex.codecogs.com/gif.latex?\inline&space;B^{Bern}&space;=0"/>), the functions in folder **logistic-normal-constant-q **can be used, while for presentation simplicity we will omit this special case.




***Generate data:***




Consider the same setting as the multinomial example above, and the standard deviation for the Gaussian noise in logistic-normal model is set as <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma&space;=1"/>.



```matlab:Code
addpath('logistic-normal-varying-q')
M=10;K=2;rho=1;
rng(2578);
A=zeros(M,K,M,K);
for i=1:M
    connect=randsample(M,rho);
    for k=1:size(connect)
     A(i,:,connect(k),:)=(rand(K,K)-0.5)*4;
    end
end
prob=0.8;
nu=zeros(M,K);nu(:,K)=ones(M,1)*log(prob/(1-prob));sigma=1;
```



Then we generate logistic-normal time series data <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;X^t&space;\rbrace_{t=0}^T"/> with sample size <img src="https://latex.codecogs.com/gif.latex?\inline&space;T=1000"/>.



```matlab:Code
T=1000;
X=data_gen_LN_Bern(M,K,T,A,nu,prob,sigma);
```



***Cross validation:***




Specify the list of tuning parameters to choose from, including a list of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha"/> and a list of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda"/>.



```matlab:Code
init_nu=zeros(M,K);
init_A=zeros(M,K,M,K);
eta=2;tol=0.0001;iter=500;
```



Choose <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\max&space;}"/>:



```matlab:Code
lambda_max=0.01;stop=false;alpha_list=[0.3 0.5 0.7];
for i=1:length(alpha_list)
    while ~stop
        lambda=lambda_max*K*sqrt(log(M)/T);
        [Ah,nu_h,~,~,~,~]=fit_LN_Bern(X,lambda,alpha_list(i),true,init_A, init_nu, eta,tol,iter);
        if sum(Ah(:).^2)>0
            lambda_max=lambda_max*2;
            init_A=Ah;init_nu=nu_h;
        else
            stop=true;
        end
    end
end
```



Choose <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda_{\min&space;}"/>:



```matlab:Code
lambda_min=lambda_max*0.001;
```



Generate evenly spaced 20 tuning parameters under the log-scale:



```matlab:Code
lambda_c_list=exp(linspace(log(lambda_min),log(lambda_max),20));
```



Perform 5-fold cross validation to choose <img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda"/>.



```matlab:Code
pred_err=zeros(length(alpha_list),length(lambda_c_list));
for i=1:length(alpha_list)
    for j=1:length(lambda_c_list)
        output=cv_LN_Bern(X,lambda_c_list(j),alpha_list(i),true,init_A,init_nu,eta,tol,iter);
        pred_err(i,j)=mean(cell2mat(output.pred_err));%
    end
end
[I,J]=find(pred_err==min(pred_err(:)));alpha=alpha_list(I);lambda=lambda_c_list(J)*K*sqrt(log(M)/T);
```



***Estimate the parameters:***



```matlab:Code
[Ah,nu_h,~,~,~,~]=fit_LN_Bern(X,lambda,alpha,true,init_A, init_nu, eta,tol,iter);
```



Estimation error:



```matlab:Code
sqrt(sum((Ah(:)-A(:)).^2))
```


```text:Output
ans = 2.5604
```


```matlab:Code
sqrt(sum((nu_h(:)-nu(:)).^2))
```


```text:Output
ans = 0.8027
```

# Mixture method


Assume that some nodes follow the multinomial autoregressive model while others follow the logistic-normal autoregressive model with time-varying event probability. Furthermore, we assume that the true categories of events associated with multinomial nodes are not accurately observed, but contaminated to be logistic-normal random vectors lying on a simplex. This is the setting of the synthetic toy model in our paper that mimics the real data behavior, and more details are included in the paper. We will show in the following how to identify the type of each node (multinomial or logistic-normal), and how to estimate the network parameter based on the their types. The required functions iare included n the folder **mixture.**




***Generate data:***




Consider the same setting as the two examples above, with first 5 nodes being logistic-normal nodes (`LN_list`) and last 5 nodes (`MN_list`) being multinomial nodes.



```matlab:Code
addpath('mixture')
M=10;K=2;rho=1;
rng(2578);
A_mix=zeros(M,K,M,K);
for i=1:M
    connect=randsample(M,rho);
    for k=1:size(connect)
     A_mix(i,:,connect(k),:)=(rand(K,K)-0.5)*4;
    end
end
LN_list=1:5;MN_list=6:10;
nu_mix=zeros(M,K);prob=0.8;
nu_mix(MN_list,:)=ones(length(MN_list),K)*log(prob/K/(1-prob));
```



First we generate the clean mixture data with sample size <img src="https://latex.codecogs.com/gif.latex?\inline&space;T=1000"/>.



```matlab:Code
T=1000;sigma_LN=1;
X=data_gen_mixed(M,K,T,A_mix,nu_mix,prob,sigma_LN,LN_list);
```



Then we contaminate the data associated to multinomial nodes to be logistic-normally distributed random vectors.



```matlab:Code
a=1;sigma_MN=0.3;
X_contam=contaminate_MN_nodes(X,MN_list,a,sigma_MN);
```



***Identify the type of each node:***




Given the contaminated data `X_contam`, we use the function `testing.m` to identify the type of each node, which outputs a test statistic `logLR`. Since we need to fit two models based on the contaminated data in `testing.m`, cross validation is also needed. The input `nlambda` is the number of potential tuning parameters to conider in cross validation. We can then estimate the set of logistic-normal nodes to be the nodes with test statistic `logLR `equal to `-Inf.`



```matlab:Code
nlambda=20;
[logLR,~,~,~,~,~,~]=testing(X_contam,nlambda);
LN_list_hat=find(logLR==-Inf);MN_list_hat=find(logLR~=-Inf);
```



Given the estimated node types, we can preprocess the contaminated data by rounding the event data associated with multinomial nodes.



```matlab:Code
X_mix=X_contam;
for m=1:length(MN_list_hat)
    for t=1:(T+1)
        if max(X_contam(t,m,:))>0
           [~,I]=sort(X_contam(t,m,:),'descend');
           X_mix(t,m,I(1))=1;
        end
    end
end
```



***Cross validation for fitting the mixture model:***




To fit the mixture model based on preprocesse data `X_mix,` three tuning parameters are required: <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda^{LN}"/>, <img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda^{MN}"/>. The first two are determined by cross validation on the logistic-normal nodes, while the last one is determinied by cross validation on the multinomial node.



```matlab:Code
init_nu=zeros(M,K);
init_A=zeros(M,K,M,K);
eta=2;tol=0.0001;iter=500;
```



Cross validation for logistic-normal nodes:



```matlab:Code
lambda_c_LN_max=0.01;stop=false;alpha_list=[0.3 0.5 0.7];
for i=1:length(alpha_list)
    while ~stop
        lambda=lambda_c_LN_max*K*sqrt(log(M)/T);
        Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
        for m=LN_list_hat
            [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_LN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),lambda,alpha_list(i),true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
        end
        if sum(Ah(:).^2)>0
            lambda_c_LN_max=lambda_c_LN_max*2;
            init_A=Ah;init_nu=nu_h;
        else
            stop=true;
        end
    end
end
lambda_c_LN_min=lambda_c_LN_max*0.001;
lambda_c_LN_list=exp(linspace(log(lambda_c_LN_min),log(lambda_c_LN_max),20));
pred_err_LN=zeros(length(LN_list_hat),length(alpha_list),length(lambda_c_LN_list));
for i=1:length(LN_list_hat)
    m=LN_list_hat(i);
    for j1=1:length(alpha_list)
        for j2=1:length(lambda_c_LN_list)
            output=cv_LN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),lambda_c_LN_list(j2),alpha_list(j1),true,init_A(m,:,:,:),init_nu(m,:),eta,tol,iter);
            pred_err_LN(i,j1,j2)=mean(cell2mat(output.pred_err));  
            Ah_temp=zeros(1,K,M,K);nu_h_temp=zeros(1,K);
            for k=1:5
                Ah_temp(1,:,:,:)=Ah_temp(1,:,:,:)+output.Ah{k};
                nu_h_temp(1,:)=nu_h_temp(1,:)+output.nu_h{k};
            end
            init_A(m,:,:,:)=Ah_temp/5;init_nu(m,:)=nu_h_temp/5;
        end
    end
end
pred_err_LN=reshape(mean(pred_err_LN,1),length(alpha_list),length(lambda_c_LN_list));
[I,J]=find(pred_err_LN==min(pred_err_LN(:)));
LN_alpha=alpha_list(I);LN_lambda_c=lambda_c_LN_list(J);
```



Cross validation for MN nodes



```matlab:Code
lambda_c_MN_max=0.01;stop=false;
while ~stop
    lambda=lambda_c_MN_max*K*sqrt(log(M)/T);
    Ah=zeros(M,K,M,K);nu_h=zeros(M,K);
    for m=MN_list_hat
        [Ah(m,:,:,:),nu_h(m,:),~,~,~,~]=fit_MN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),lambda,true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
    end
    if sum(Ah(:).^2)>0
        lambda_c_MN_max=lambda_c_MN_max*2;
        init_A=Ah;init_nu=nu_h;
    else
        stop=true;
    end
end
lambda_c_MN_min=lambda_c_MN_max*0.001;
lambda_c_MN_list=exp(linspace(log(lambda_c_MN_min),log(lambda_c_MN_max),20));
pred_err_MN=zeros(length(MN_list_hat),length(lambda_c_MN_list));
for i=1:length(MN_list_hat)
    m=MN_list_hat(i);
    for j=1:length(lambda_c_MN_list)
        output=cv_MN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),lambda_c_MN_list(j),true,init_A(m,:,:,:),init_nu(m,:),eta,tol,iter);
        pred_err_MN(i,j)=mean(cell2mat(output.pct_err));  
        Ah_temp=zeros(1,K,M,K);nu_h_temp=zeros(1,K);
        for k=1:5
            Ah_temp(1,:,:,:)=Ah_temp(1,:,:,:)+output.Ah{k};
            nu_h_temp(1,:)=nu_h_temp(1,:)+output.nu_h{k};
        end
        init_A(m,:,:,:)=Ah_temp/5;init_nu(m,:)=nu_h_temp/5;
    end
end
[~,ind]=min(mean(pred_err_MN,1)); 
MN_lambda_c=lambda_c_MN_list(ind);
```



***Estimate the parameters:***



```matlab:Code
LN_lambda=LN_lambda_c*K*sqrt(log(M)/T);
MN_lambda=MN_lambda_c*K*sqrt(log(M)/T);
Ah_mix=zeros(M,K,M,K);nu_h_mix=zeros(M,K);
init_A=zeros(M,K,M,K);init_nu=zeros(M,K);
for m=1:M
    if sum(LN_list_hat==m)>0
        [Ah_mix(m,:,:,:),nu_h_mix(m,:),~,~,~,~]=fit_LN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),LN_lambda,LN_alpha,true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
    else
        [Ah_mix(m,:,:,:),nu_h_mix(m,:),~,~,~,~]=fit_MN_node(X_mix(1:T,:,:),X_mix(2:(T+1),m,:),MN_lambda,true,init_A(m,:,:,:), init_nu(m,:), eta,tol,iter);
    end
end
```



Estimation error:



```matlab:Code
sqrt(sum((Ah_mix(:)-A_mix(:)).^2))
```


```text:Output
ans = 5.0571
```


```matlab:Code
sqrt(sum((nu_h_mix(:)-nu_mix(:)).^2))
```


```text:Output
ans = 0.9574
```

  

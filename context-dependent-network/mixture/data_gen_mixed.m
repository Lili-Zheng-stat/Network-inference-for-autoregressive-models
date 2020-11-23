function X=data_gen_mixed(M,K,T,A_mixed,nu_mixed,prob,sigma_LN,set_LN)
%generate time series data X from mixture model with A_mixed and nu_mixed.
%set_LN specifies the nodes following logistic-normal model.
%prob is the initial probability (constant) for each user to tweet, sigma is gaussian
%noise variance


%initial state
X =zeros(1,M,K);
for m=1:M 
    X(1,m,randsample(K,1))=rand(1)<=prob;
end 

for t=1:T
    X_temp=zeros(1,M,K);
    for m=1:M
        if sum(set_LN==m)>0
            q=1/(1+exp(-sum(reshape(A_mixed(m,K,:),1,M*K).*X(t,:))-nu_mixed(m,K)));
            if(rand(1)<=q)
              Y=ones(1,K);
              for i=1:(K-1)
                Y(1,i)=exp(sum(reshape(A_mixed(m,i,:),1,M*K).*X(t,:))+nu_mixed(m,i)+random('norm',0,sigma_LN));
              end
              X_temp(1,m,:)=Y/sum(Y);
            end
        else
            prob_temp=ones(1,K+1);
            for i=1:K
                 prob_temp(1,i)=exp(sum(reshape(A_mixed(m,i,:,:),1,M*K).*reshape(X(t,:,:),1,M*K))...
                    +nu_mixed(m,i));
            end
            prob_temp=prob_temp/sum(prob_temp);
            Y=mnrnd(1,prob_temp);
            X_temp(1,m,:)=Y(1:K);
        end
    end
    X=cat(1,X,X_temp);
end
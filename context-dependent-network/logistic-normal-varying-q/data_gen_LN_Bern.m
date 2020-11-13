function X=data_gen_LN_Bern(M,K,T,A,nu,prob,sigma)
%generate data X from logistic-normal model with time varying q

  %initial state, with no mixed membership (random category)
  X =zeros(1,M,K);
  for m=1:M 
      X(1,m,randsample(K,1))=rand(1)<=prob;
  end
  
  for t=1:T
      X_temp=zeros(1,M,K);
      q=zeros(1,M);
      for m=1:M
          q(1,m)=1/(1+exp(-sum(reshape(A(m,K,:),1,M*K).*X(t,:))-nu(m,K)));
          if(rand(1)<=q(1,m))
              Y_temp=ones(1,K);
              for i=1:(K-1)
                Y_temp(1,i)=exp((sum(reshape(A(m,i,:),1,M*K).*X(t,:))+nu(m,i)+random('norm',0,sigma)));
              end
              X_temp(1,m,:)=Y_temp/sum(Y_temp);
          end
      end
      X=cat(1,X,X_temp);
  end
  
      
function X=data_gen_LN(M,K,T,A,nu,prob,q,sigma)
%generate logistic-normal data X, when event probability is a constant vector q

  X =zeros(1,M,K);
  for i=1:M 
      X(1,i,randsample(K,1))=rand(1)<=prob;
  end
  %initial state, with no mixed membership (random category)
  
  for t=1:T
      X_temp=zeros(1,M,K);
      for m=1:M
          if(rand(1)<=q(1,m))
              X_temp=ones(1,K);
              for i=1:(K-1)
                X_temp(1,i)=exp((sum(reshape(A(m,i,:),1,M*K).*X(t,:))+nu(m,i)+random('norm',0,sigma)));
              end
              X_temp(1,m,:)=X_temp/sum(X_temp);
          end
      end
      X=cat(1,X,X_temp);
  end
  
      
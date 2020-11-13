function X=data_gen_MN(M,K,T,A,nu,prob)
%generate multinomial data X

  %initial state
  X =zeros(1,M,K);
  for m=1:M
      X(1,m,randsample(K,1))=(rand(1)<=prob);
  end
  
  for t=1:T
      X_temp=zeros(1,M,K);
      for m=1:M
          prob_temp=ones(1,K+1);
          for i=1:K
              prob_temp(1,i)=exp(sum(reshape(A(m,i,:,:),1,M*K).*reshape(X(t,:,:),1,M*K))...
                  +nu(m,i));
          end
          prob_temp=prob_temp/sum(prob_temp);
          q=mnrnd(1,prob_temp);
          X_temp(1,m,:)=q(1:K);
      end
      X=cat(1,X,X_temp);
  end
  
      
  

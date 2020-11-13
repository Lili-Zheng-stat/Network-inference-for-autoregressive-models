function loss=calc_loss_LN(A,nu,X)
%calculte the log-likelihood loss function at A and nu,
%with data Q. 
[T,M,K]=size(X);
T=T-1;
loss=0;
for t=1:T
    for m=1:M
        if sum(X(t+1,m,:))~=0
            Y=zeros(1,K-1);mu=zeros(1,K-1);
            for i=1:(K-1)
                Y(i)=log(X(t+1,m,i)/X(t+1,m,K));
                mu(i)=(sum(reshape(A(m,i,:),1,M*K).*X(t,:))+nu(m,i));
            end
            loss=loss+sum((Y-mu).^2)/2;
        end
    end
end
loss=loss/T;
end
function [grad_A, grad_nu]=calc_grad_LN(A,nu,X)
%calculate gradient w.r.t A and nu with data X. 
[T,M,K]=size(X);
T=T-1;
grad_A=zeros(M,K-1,M,K);
grad_nu=zeros(M,K-1);
ind=sum(reshape(X(2:(T+1),:,:),T,M,K),3)>0;
Y=ind.*log(X(2:(T+1),:,1:(K-1))./X(2:(T+1),:,K));
Y(isnan(Y))=0;
for m=1:M
    for i=1:(K-1)
        term1=-sum(Y(:,m,i).*ind(:,m).*X(1:T,:,:),1)/T;
        term2=sum(ind(:,m).*(sum(X(1:T,:).*reshape(A(m,i,:),1,M*K),2)+nu(m,i))...
            .*X(1:T,:,:),1)/T;
        grad_A(m,i,:,:)=term1+term2;
        term3=-sum(Y(:,m,i).*ind(:,m),1)/T;
        term4=sum(ind(:,m).*(sum(X(1:T,:).*reshape(A(m,i,:),1,M*K),2)+nu(m,i)),1)/T;
        grad_nu(m,i)=term3+term4;
    end
end


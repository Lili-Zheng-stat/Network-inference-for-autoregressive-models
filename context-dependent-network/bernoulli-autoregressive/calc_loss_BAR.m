function loss=calc_loss_BAR(A,nu,Y)
%calculte the multinomial log-likelihood loss function at A (M*p*M*p), intercept nu (M*p),
%and data Q. 

[T,M]=size(Y);
T=T-1;
loss=0;
for t=1:T
    for m=1:M
        intensity=sum(reshape(A(m,:),1,M).*Y(t,:))+nu(m);
        loss=loss+log(1+exp(intensity))-intensity*Y(t+1,m);
    end
end
loss=loss/T;
end
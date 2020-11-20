function [grad_A, grad_nu]=calc_grad_BAR(A,nu,Y)
%calculate multinomial gradient w.r.t A and nu with data Q. 

[T,M]=size(Y);
T=T-1;
grad_A=zeros(M,M);
grad_nu=zeros(M,1);
for t=1:T
    for m=1:M
        coef_temp=exp(sum(reshape(A(m,:),1,M).*Y(t,:))+nu(m,1));
        upd=coef_temp/(coef_temp+1)-Y(t+1,m);
        grad_A(m,:)=grad_A(m,:)+upd*reshape(Y(t,:),1,M);
        grad_nu(m,:)=grad_nu(m,:)+upd;
    end
end
grad_A=grad_A/T;
grad_nu=grad_nu/T;

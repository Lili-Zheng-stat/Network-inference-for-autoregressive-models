function [Ah,nu_h,loss,grad_A,grad_nu,cvg]=fit_mult_sep(Q_cov,Q_resp,lambda, ...
intercept,init_A, init_nu, eta,tol,iter)
%estimation under multinomial GLM, with covariates Q_cov, response Q_resp. Q_resp is rounded data. 
%projected gradient descent algorithm to solve Ah (M*p*M*p) and nu_h (M*p), based on
%Q_cov, Q_resp, penalty parameter lambda
%If intercept is true, estimate nu; o.w. assume nu to be 0
%loss_true is the loss (with penalty added) evaluated at the true A^*, L2E are L2 distance
%between Ah and A^*, nu_h and nu^*.
%If intercept is false, nu_h L2E_nu are all zeros
%loss and L2E's are iter dimensional (trace of estimation) (L2E_nu is scalar
%if intercept is false)

%initialization
[~,M,p]=size(Q_cov);     
if intercept
    nu_h=init_nu;
else
    nu_h=zeros(M,p);
end
Ah=init_A;
kk=0;
chg_per=Inf;

%initialize loss & L2E
loss=calc_loss_mult_sep(Ah,nu_h,Q_cov,Q_resp)+lambda*nmR(Ah);


while kk<iter && chg_per>tol
    [grad_A, grad_nu]=calc_grad_mult_sep(Ah,nu_h,Q_cov,Q_resp);
    %determine initial step size^(-1)
    if kk==0
        ss=1;
    elseif intercept
        inprod1=(nu_h-nu_h_prev).*(grad_nu-grad_nu_prev);
        inprod2=(Ah-Ah_prev).*(grad_A-grad_A_prev);
        num=sum(inprod1(:))+sum(inprod2(:));
        denom=(norm(nu_h-nu_h_prev,'fro'))^2+(norm(reshape(Ah-Ah_prev,M*p,M*p),'fro'))^2;
        ss=num/denom;
    else
        inprod=(Ah-Ah_prev).*(grad_A-grad_A_prev);
        num=sum(inprod(:));
        denom=(norm(reshape(Ah-Ah_prev,M*p,M*p),'fro'))^2;
        ss=num/denom;
    end
    
    accept=false;
    while ~accept
        if intercept
            grad_nu_upd=nu_h-grad_nu/ss;
        end
        grad_A_upd=Ah-grad_A/ss;
        
        %soft-thresholding
        Ah_temp=zeros(M,p,M,p);
        for i=1:M
            for j=1:M
                nm=norm(reshape(grad_A_upd(i,:,j,:),p,p),'fro');
                if nm>lambda/ss
                   Ah_temp(i,:,j,:)= grad_A_upd(i,:,j,:)*(nm-lambda/ss)/nm;
                end
            end
        end
        if intercept
            loss_temp=calc_loss_mult_sep(Ah_temp,grad_nu_upd,Q_cov,Q_resp)+lambda*nmR(Ah_temp);
        else
            loss_temp=calc_loss_mult_sep(Ah_temp,nu_h,Q_cov,Q_resp)+lambda*nmR(Ah_temp);
        end
       
        %check acceptance: non-increase in loss
        if kk==0
            accept=(loss_temp<=loss)||ss>=1000;
        else
            accept=(loss_temp<=loss(kk+1))||ss>=1000;
        end
        ss=eta*ss;
    end
    
    if intercept
        nu_h_prev=nu_h;
        nu_h=grad_nu_upd;
        grad_nu_prev=grad_nu;
    end
    Ah_prev=Ah;
    Ah=Ah_temp;
    grad_A_prev=grad_A;
    kk=kk+1;
    loss=cat(1,loss,loss_temp);
    
    
    %check convergence
    if intercept
        nm_prev=sqrt((norm(nu_h_prev,'fro'))^2+(norm(reshape(Ah_prev,M*p,M*p),'fro'))^2);
        nm_diff=sqrt((norm(nu_h-nu_h_prev,'fro'))^2+(norm(reshape(Ah-Ah_prev,M*p,M*p),'fro'))^2);
    else
        nm_prev=norm(reshape(Ah_prev,M*p,M*p),'fro');
        nm_diff=norm(reshape(Ah-Ah_prev,M*p,M*p),'fro');
    end
    
    if nm_diff>0
        chg_per=nm_diff/nm_prev;
    else
        chg_per=nm_diff;
    end
end


if(chg_per<=tol)
    cvg=true;
else
    cvg=false;
end
end
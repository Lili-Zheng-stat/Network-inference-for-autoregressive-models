function [Ah,nu_h,loss,grad_A,grad_nu,cvg]=fit_R_alpha_sep(Q_cov,Q_resp,lambda,alpha,omega, ...
intercept,init_A, init_nu, eta,tol,iter)
%projected gradient descent algorithm to solve Ah (M*p*M*p) and nu_h (M*p), based on
%covariates Q_cov and response Q_resp,sigma, with weight parameter alpha, penalty parameter lambda
%If intercept is true, estimate nu; o.w. assume nu to be 0
%loss_true is the loss (with penalty added) evaluated at the true A^*, L2E are L2 distance
%between Ah and A^*, nu_h and nu^*.
%If intercept is false, init_nu, nu_h, L2E_nu are all zeros
%loss and L2E's are iter dimensional (trace of estimation) (L2E_nu is scalar
%if intercept is false)

%initialization
[~,M,p]=size(Q_cov);
nu_h=init_nu;
Ah=init_A;
kk=0;
chg_per=Inf;

%initialize loss & L2E
loss=calc_loss_GSM_sep(Ah,nu_h,Q_cov,Q_resp,alpha)+lambda*nmR_weighted(Ah,alpha,omega);


while kk<iter && chg_per>tol
    [grad_A, grad_nu]=calc_grad_GSM_sep(Ah,nu_h,Q_cov,Q_resp);
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
    if isnan(ss)||ss<=0
        ss=1;
    else
        ss=min(100,max(ss,0.0001));
    end
    
    accept=false;
    while ~accept
        grad_A_upd=Ah-grad_A/ss;
        
        %soft-thresholding
        Ah_temp=zeros(M,p,M,p);
        for i=1:M
            for j=1:M
                nm=sqrt(alpha*norm(reshape(grad_A_upd(i,1:(p-1),j,:),p-1,p),'fro')^2+...
                    omega*(1-alpha)*norm(reshape(grad_A_upd(i,p,j,:),1,p),'fro')^2);
                if nm>lambda/ss
                   Ah_temp(i,:,j,:)= grad_A_upd(i,:,j,:)*(nm-lambda/ss)/nm;
                end
            end
        end
        
        if intercept
            grad_nu_upd=nu_h-grad_nu/ss;
        end
        if intercept
            loss_temp=calc_loss_GSM_sep(Ah_temp,grad_nu_upd,Q_cov,Q_resp,alpha)+lambda*nmR_weighted(Ah_temp,alpha,omega);
        else
            loss_temp=calc_loss_GSM_sep(Ah_temp,nu_h,Q_cov,Q_resp,alpha)+lambda*nmR_weighted(Ah_temp,alpha,omega);
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
    
    if nm_prev>0
        chg_per=nm_diff/min(nm_prev,10);
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
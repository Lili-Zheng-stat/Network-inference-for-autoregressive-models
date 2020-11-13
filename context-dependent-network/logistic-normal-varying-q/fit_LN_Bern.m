function [Ah,nu_h,loss,grad_A,grad_nu,cvg]=fit_LN_Bern(X,lambda,alpha,...
intercept,init_A, init_nu, eta,tol,iter)
%Ah and nu_h are estimated network parameter and offset parameter; loss is
%a vector containing the penalized log-likeliihood loss evaluated at each
%iteration; grad_A and grad_nu are the gradients w.r.t. A and nu when the
%algorithm stops; cvg = true if tolerance is met when algorithm stops.

%initialization
[T,M,K]=size(X);
T=T-1;
nu_h=init_nu;
Ah=init_A;
kk=0;
chg_per=Inf;

%initialize loss & L2E
loss=calc_loss_LN_Bern(Ah,nu_h,X,alpha)+lambda*nmR_weighted(Ah,alpha);

while kk<iter && chg_per>tol
    [grad_A, grad_nu]=calc_grad_LN_Bern(Ah,nu_h,X);
    if alpha==0
        grad_A(:,1:(K-1),:,:)=0;
        grad_nu(:,1:(K-1))=0;
    elseif alpha==1
        grad_A(:,K,:,:)=0;
        grad_nu(:,K)=0;
    end
    %determine initial step size^(-1)
    if kk==0
        ss=1;
    elseif intercept
        inprod1=(nu_h-nu_h_prev).*(grad_nu-grad_nu_prev);
        inprod2=(Ah-Ah_prev).*(grad_A-grad_A_prev);
        num=sum(inprod1(:))+sum(inprod2(:));
        denom=(norm(nu_h-nu_h_prev,'fro'))^2+(norm(reshape(Ah-Ah_prev,M*K,M*K),'fro'))^2;
        ss=num/denom;
    else
        inprod=(Ah-Ah_prev).*(grad_A-grad_A_prev);
        num=sum(inprod(:));
        denom=(norm(reshape(Ah-Ah_prev,M*K,M*K),'fro'))^2;
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
        Ah_temp=zeros(M,K,M,K);
        for i=1:M
            for j=1:M
                nm=sqrt(alpha*norm(reshape(grad_A_upd(i,1:(K-1),j,:),K-1,K),'fro')^2+...
                    (1-alpha)*norm(reshape(grad_A_upd(i,K,j,:),1,K),'fro')^2);
                if nm>lambda/ss
                   Ah_temp(i,:,j,:)= grad_A_upd(i,:,j,:)*(nm-lambda/ss)/nm;
                end
            end
        end
        
        if intercept
            grad_nu_upd=nu_h-grad_nu/ss;
        end
        if intercept
            loss_temp=calc_loss_LN_Bern(Ah_temp,grad_nu_upd,X,alpha)+lambda*nmR_weighted(Ah_temp,alpha);
        else
            loss_temp=calc_loss_LN_Bern(Ah_temp,nu_h,X,alpha)+lambda*nmR_weighted(Ah_temp,alpha);
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
        nm_prev=sqrt((norm(nu_h_prev,'fro'))^2+(norm(reshape(Ah_prev,M*K,M*K),'fro'))^2);
        nm_diff=sqrt((norm(nu_h-nu_h_prev,'fro'))^2+(norm(reshape(Ah-Ah_prev,M*K,M*K),'fro'))^2);
    else
        nm_prev=norm(reshape(Ah_prev,M*K,M*K),'fro');
        nm_diff=norm(reshape(Ah-Ah_prev,M*K,M*K),'fro');
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
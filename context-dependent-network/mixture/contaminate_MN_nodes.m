function X_contam=contaminate_MN_nodes(X,MN_list,a,sigma_MN)
%Contaminate the events data associated with multinomial nodes (specified
%by MN_list) to be logisitc-normal random vectors, with location parameter
%a and scale parameter sigma_MN
X_contam=X;
[T,~,K]=size(X);T=T-1;
for t=2:(T+1)
    for i=1:length(MN_list)
        if sum(X(t,MN_list(i),:))>0
           %add some noise to the one-hot vector
           X_temp=ones(1,K);
           Z=X(t,MN_list(i),:);
           if X(t,MN_list(i),K)==1
              mu=-ones(1,K-1)*a;
           else
              mu=zeros(1,K-1);
              mu(X(t,MN_list(i),1:(K-1))==1)=a;
           end
           X_temp(1:(K-1))=exp(mu+normrnd(0,sigma_MN,1,K-1));
           X_contam(t,MN_list(i),:)=X_temp/sum(X_temp);
        end
    end 
end

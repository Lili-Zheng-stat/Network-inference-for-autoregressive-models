function nm=nmR_weighted(A,alpha)
%calculate R norm
M=size(A,1);
p1=size(A,2);p2=size(A,4);
nm=0;
for i=1:M
    for j=1:M
        nm=nm+sqrt(alpha*norm(reshape(A(i,1:(p1-1),j,:),p1-1,p2),'fro')^2+...
            (1-alpha)*norm(reshape(A(i,p1,j,:),1,p2),'fro')^2);
    end
end
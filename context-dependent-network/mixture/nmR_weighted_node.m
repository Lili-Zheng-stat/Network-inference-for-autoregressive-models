function nm=nmR_weighted_node(A,alpha)
%calculate R norm
M=size(A,3);
p1=size(A,2);p2=size(A,4);
nm=0;
for j=1:M
    nm=nm+sqrt(alpha*norm(reshape(A(1,1:(p1-1),j,:),p1-1,p2),'fro')^2+...
        (1-alpha)*norm(reshape(A(1,p1,j,:),1,p2),'fro')^2);
end
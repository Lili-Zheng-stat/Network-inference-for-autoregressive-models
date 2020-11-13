function nm=nmR_node(A)
%calculate R norm
M=size(A,3);
p1=size(A,2);p2=size(A,4);
nm=0;
for j=1:M
    nm=nm+norm(reshape(A(1,:,j,:),p1,p2),'fro');
end
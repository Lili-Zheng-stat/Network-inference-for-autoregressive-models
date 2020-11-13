function nm=nmR(A)
%calculate R norm
M=size(A,1);
p1=size(A,2);p2=size(A,4);
nm=0;
for i=1:M
    for j=1:M
        nm=nm+norm(reshape(A(i,:,j,:),p1,p2),'fro');
    end
end
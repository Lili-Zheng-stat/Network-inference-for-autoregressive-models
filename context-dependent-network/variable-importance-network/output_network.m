function g_mat=output_network(A,thrs,negative)
         s=[];t=[];weight=[];type1=[];type2=[];
         [M,p1,~,p2]=size(A);
         for i=1:p2
             for j=1:p1
                 if negative
                    [row,col]=find(reshape(abs(A(:,j,:,i)),M,M)>thrs);
                 else
                    [row,col]=find(reshape(A(:,j,:,i),M,M)>thrs);
                 end
                 ind=find(row~=col);
                 row=row(ind);
                 col=col(ind);
                 v=zeros(length(row),1);
                for l=1:length(row)
                    v(l)=A(row(l),j,col(l),i);
                end
                t=cat(1,t,row);
                s=cat(1,s,col);
                weight=cat(1,weight,v);
                type1=cat(1,type1,repmat(i,length(row),1));%source topic
                type2=cat(1,type2,repmat(j,length(row),1));%target topic
             end
         end
         g_mat=[s t weight type1 type2];
end


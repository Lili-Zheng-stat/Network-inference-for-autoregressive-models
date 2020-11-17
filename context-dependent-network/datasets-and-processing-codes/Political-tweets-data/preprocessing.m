clc
clear all
data=readtable('tweets_with_ideology_score.csv');
data=data(:,2:5);
users=unique(bert_dat.user)
%8th and 17th users are the same
bert_dat.user(string(bert_dat.user)==users(8))=users(17);

users=unique(bert_dat.user);
labels = ['rep';'dem';'rep';'rep';'rep';'rep';'rep';'dem';'dem';...
    'rep';'rep';'rep';'rep';'rep';'dem';'rep';'rep';'dem';'rep';'rep';'rep';...
    'dem';'dem'];
labels=string(labels);

min(data.weight_dem)
min(data.weight_rep)
%too close to 0,1. rescale;different loss;

data.weight_dem=0.5+(bert_dat.weight_dem-0.5)*199/200;
data.weight_rep=0.5+(bert_dat.weight_rep-0.5)*199/200;

%construct data X
min_t=min(data.time_unix);
max_t=max(data.time_unix)+1;
window_t=(max_t-min_t)/1000;%around 3.7 hours
%window_t=8*3600;
T=999;M=23;p=2;
X=zeros(T+1,M,p);
for t=1:(T+1)
    L=min_t+(t-1)*window_t;
    U=min_t+t*window_t;
    ind=find(bert_dat.time_unix>=L&bert_dat.time_unix<U);
    for m=1:M
        ind_ind=find(string(bert_dat.user(ind))==users(m));
        if ~isempty(ind_ind)
            X(t,m,1)=sum(bert_dat.weight_dem(ind(ind_ind)))/length(ind_ind);
            X(t,m,2)=1-X(t,m,1);
        end
    end
end
save('X.mat','X','users','labels');

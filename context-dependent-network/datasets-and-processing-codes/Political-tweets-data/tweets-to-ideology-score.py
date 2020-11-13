from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
args = get_args_parser().parse_args(['-model_dir', 'uncased_L-24_H-1024_A-16'
                                     , '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-show_tokens_to_client',
                                     '-pooling_strategy', 'NONE',
                                     '-cpu'])
server = BertServer(args)
server.start()
from bert_serving.client import BertClient
import pandas
import numpy as np
import sys
from numpy import linalg as LA
from ast import literal_eval


bc=BertClient()
dat_all=pandas.read_csv('tweets_raw_data.csv')
for argument in range(1,84):
    dat=dat_all.loc[(argument*1000+1):(argument+1)*1000,:]
    dat_new=dat.iloc[np.where(~dat.iloc[:,7].isna())[0],:]
    text=pandas.Series.tolist(dat_new.loc[:,'text'])
    vec=bc.encode(text,show_tokens=True)
    with open('embedding/token_{}.txt'.format(argument), 'w') as fp:
        fp.write('\n'.join('%s'%x for x in vec[1]))
    with open("embedding/vec_{}.txt".format(argument), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(vec[0].shape))
        for data_slice in vec[0]:  
            np.savetxt(outfile, data_slice, fmt='%-7.8f')
            outfile.write('# New slice\n');
 
''''
read embeddings
''''
token_list=[]
embedding_list=np.empty([0,1024])
for argument in range(1,84):
    with open("embedding/token_{}.txt".format(argument)) as f:
        tokenlist_temp=[list(literal_eval(line)) for line in f]
    embedding_list_temp = np.loadtxt("embedding/vec_{}.txt".format(argument))
    a=np.zeros(len(tokenlist_temp))
    for i in range(len(tokenlist_temp)): 
        a[i]=len(tokenlist_temp[i])
    embedding_list_temp=embedding_list_temp.reshape(int(embedding_list_temp.shape[0]/int(max(a))),int(max(a)),1024)
    embedding_list_temp=np.mean(embedding_list_temp,axis=1)
    embedding_list=np.append(embedding_list,embedding_list_temp,axis=0)
    token_list.extend(tokenlist_temp)

dat_new=dat_all.iloc[np.where(~dat_all.iloc[:,7].isna())[0],:]
users=np.unique(dat_new.loc[:,'user'])
labels_users =np.array(["rep","dem","rep","rep","rep","rep","rep","rep","rep","dem","dem",\
    "rep","rep","rep","rep","rep","dem","rep","rep","dem","rep","rep","rep",\
    "dem","dem"])
    
''''
train a neural network
''''
import math
from random import sample
min_t=min(dat_new.loc[:,'time_unix'])
max_t=max(dat_new.loc[:,'time_unix'])
'''use the first half of the data for training the neural network'''
ind=np.reshape(np.where(dat_new.loc[:,'time_unix']<=min_t+(max_t-min_t)/2),\
               sum(dat_new.loc[:,'time_unix']<=min_t+(max_t-min_t)/2))
train_ind=sample(list(ind),math.floor(len(ind)*0.8))
train_embedding=embedding_list[train_ind]
labels=np.zeros(len(token_list))
for i in range(len(users)):
    labels[dat_new.loc[:,'user']==users[i]]=(labels_users[i]=="rep")
''''
0 (first category) is democratic, 1 (second category) is republican 
''''
train_label=labels[train_ind]
test_ind = ~np.array([(i in train_ind) for i in ind])
test_ind=np.reshape(np.where(test_ind),sum(test_ind))
test_embedding=embedding_list[test_ind]
test_label=labels[test_ind]


import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(128,input_shape=(1024,), activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_embedding,train_label,epochs=20)   
test_loss, test_acc = model.evaluate(test_embedding, test_label)
print('Test accuracy:', test_acc) 

'''use the trained neural network to predict the ideology score of the latter half of the data'''
alg_ind=np.reshape(np.where(dat_new.loc[:,'time_unix']>min_t+(max_t-min_t)/2),\
               sum(dat_new.loc[:,'time_unix']>min_t+(max_t-min_t)/2))
alg_weights = model.predict(embedding_list[alg_ind])


''''
output data with time_unix, users and weights
''''
from pandas import DataFrame
output_dat=dat_new.iloc[alg_ind,[3,5]]
output_dat['weight_dem']=alg_weights[:,0]
output_dat['weight_rep']=alg_weights[:,1]
output_dat.to_csv('bert_output.csv')


    
    

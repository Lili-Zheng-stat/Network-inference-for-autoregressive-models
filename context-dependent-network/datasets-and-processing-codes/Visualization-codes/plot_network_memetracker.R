library(igraph)
M=58;

color_list=c(rgb(0,128/255,1),rgb(0,0,0),rgb(1,102/255,102/255),
             rgb(0,153/255,0),rgb(1,1,102/255))
name_list=c('43things','alertnet','amazon','baycitytribune','bizjournals',
            'bostonherald','breitbart','ca.rd.yahoo','canadianbusiness','chambliss.senate',
            'channelnewsasia','cnn','dailyherald','daytondailynews','denverpost',
            'dnaindia','earthtimes','gothamistllc','fixya','forbes','forum.prisonplanet',
            'forums.g4tv','golivewire','gotpoetry','LA.craigslist','marketwatch',
            'news.cnet','news.scotsman','newsday','newsmeat','newsobserver',
            'northjersey','open.salon','philly','post-gazette','pr-inside','prnewswire',
            'reuters','ebay','seekingalpha','sltrib','sott','sports.espn',
            'statesman','thefacts','theolympian','timesofindia.indiatimes','topnews.in',
            'uk.answers.yahoo','uk.news.yahoo','uk.reuters','usatoday','videohelp',
            'washingtonpost','wbay','webwire','wral','yelp')

sub_mat=function(g_mat,nnode_show,thrs){
  #normalize edge weights
  g_mat[,3]=g_mat[,3]/max(abs(g_mat[,3]));
  g_mat=g_mat[abs(g_mat[,3])>thrs,];
  #choose neighbors
  nb_list=unique(g_mat$V1);
  nb_strength=rep(0,length(nb_list))
  for (i in 1:length(nb_list)){
    nb_strength[i]=sum(abs(g_mat$V3[g_mat$V1==nb_list[i]]))
  }
  nnode_show=min(nnode_show,length(nb_list));
  nb_list=nb_list[order(-nb_strength)[1:nnode_show]];
  output=list(g_mat,nb_list)
  return(output)
}
mat_to_graph=function(g_mat,nb_list,name_list){
  #define graphs (nodes, edges, types, names, weights)
  g_mat_nb=NULL;
  for(i in 1:length(nb_list)){
    g_mat_nb=rbind(g_mat_nb,g_mat[g_mat$V1==nb_list[i],] )
  }
  node_list=unique(c(g_mat_nb$V2,nb_list));
  graph=graph_from_data_frame(d=g_mat_nb, vertices=node_list, directed=T);
  V(graph)$name=name_list[node_list];
  E(graph)$weight=g_mat_nb[,3];
  E(graph)$topic=g_mat_nb$V4;
  E(graph)$type=(E(graph)$weight>0);
  return(graph);
}
plot_graph=function(graph,edgescale,arrowscale, color_list,layout){
  #plot graph with topic, weight, type.
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape='circle',vertex.size=8,
       vertex.label.dist=2)
  for (e in 1:ecount(graph)) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2,edge.color=color_list[E(graph)$topic[e]],edge.lty=ifelse(E(graph)$type[e],1,2),
         edge.curved=E(graph)$topic[e]*0.2,edge.width=abs(E(graph)$weight[e])*edgescale,
         edge.arrow.size=min(1.5,abs(E(graph)$weight[e])*arrowscale), layout=layout, vertex.shape="none",
         vertex.label=NA, add=TRUE)
  }
}

#check with specific edges
g_mat_varimportance_MN=read.csv('meme_MN_infl_graph.csv',header=F);
g_mat_varimportance_LN=read.csv('meme_LN_infl_graph.csv',header=F);
g_mat_combined_2tuning_varimportance=read.csv(sprintf("meme_infl_absolute_graph_mod_2tuning.csv"),header=F)
ind=which(g_mat_varimportance_MN[,1]==which(name_list=='alertnet')&g_mat_varimportance_MN[,2]==which(name_list=='reuters'))
g_mat_varimportance_MN[ind,]


#same neighbors, star-shaped. MN absolute, relative; LN relative, overall; variable importance (MN, LN)
g_mat_MN=read.csv('meme_MN_graph.csv',header=F);
g_mat_LN=read.csv('meme_LN_graph.csv',header=F);
g_mat_mixed=read.csv(sprintf("meme_mixed_graph.csv"),header=F)

nnode_show=8;thrs_param=0.2;thrs=0.1;edgescale=5;arrowscale=1.5;
for(m in 1:M){
  #select subnetwork (edges pointing to node m)
  g_mat_MN_sub=g_mat_MN[g_mat_MN$V2==m&g_mat_MN$V4==g_mat_MN$V5,]
  g_mat_LN_sub=g_mat_LN[g_mat_LN$V2==m&g_mat_LN$V4==g_mat_LN$V5,]
  g_mat_mixed_sub=g_mat_mixed[(g_mat_mixed$V2==m)&g_mat_mixed$V4==g_mat_mixed$V5,];
  
  #for each subnetwork, filter out the top nodes and edges with parameters higher than thrs_param
  output_MN=sub_mat(g_mat_MN_sub,nnode_show,thrs);
  output_LN=sub_mat(g_mat_LN_sub,nnode_show,thrs);
  output_mixed=sub_mat(g_mat_mixed_sub,nnode_show,thrs);
  
  #create nb_list as a union for all top neighbors, then create graphs
  nb_list=unique(c(output_MN[[2]],output_LN[[2]],output_mixed[[2]]));
  
  #create graph 
  graph_MN=mat_to_graph(output_MN[[1]],nb_list,name_list);
  graph_LN=mat_to_graph(output_LN[[1]],nb_list,name_list);
  graph_mixed=mat_to_graph(output_mixed[[1]],nb_list,name_list);
  
  #plot graphs with the same layout
  par(mar=c(0,0.5,0,2))
  l=layout_as_star(graph_MN, center = V(graph_MN)[names(V(graph_MN))==name_list[m]], order = NULL)
  png(sprintf("fig/meme_LN_varimportance_subnetwork_%d.png",m))
  plot_graph(graph_LN,edgescale,arrowscale,color_list,l)
  dev.off()
  png(sprintf("fig/meme_MN_varimportance_subnetwork_%d.png",m))
  plot_graph(graph_MN,edgescale,arrowscale,color_list,l)
  dev.off()
  png(sprintf("fig/meme_mixed_varimportance_subnetwork_%d.png",m))
  plot_graph(graph_mixed,edgescale,arrowscale,color_list,l)
  dev.off()
} 


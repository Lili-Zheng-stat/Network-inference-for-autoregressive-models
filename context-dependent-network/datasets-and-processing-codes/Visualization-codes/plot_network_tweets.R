#twitter data network after fitting each node based on the test
library(igraph)
M=23;
color_list=c("blue","red")
party_list=c("rep","dem","rep","rep","rep","rep","rep","dem",
             "dem","rep","rep","rep","rep","rep","dem",
             "rep","rep","dem","rep","rep","rep","dem","dem");
name_list=c("Ben & Candy Carson","Bernie Sanders","Carly Fiorina",
            "Donald J. Trump","GOP","Gov. Mike Huckabee","Governor Christie",
            "Hillary Clinton","House Democrats",
            "House Republicans","Jeb Bush","Jim Gilmore","John Kasich",
            "Marco Rubio","Martin O'Malley","Mike Pence",
            "Rick Santorum","Senate Democrats", "Senate Republicans",
            "Senator Rand Paul","Ted Cruz","The Democrats","Tim Kaine");
party_vec=c('dem','rep')

mat_to_graph=function(g_mat,name_list,party_list,thrs){
  #normalize edge weights
  g_mat[,3]=g_mat[,3]/max(abs(g_mat[,3]));
  g_mat=g_mat[abs(g_mat[,3])>thrs,]
  #define graphs (nodes, edges, types, names, weights)
  graph=graph_from_data_frame(d=g_mat, vertices=1:M, directed=T);
  V(graph)$type=party_list;
  V(graph)$name=name_list;
  E(graph)$weight=g_mat[,3];
  E(graph)$type=(E(graph)$weight>0)
  E(graph)$type1=g_mat[,4];
  E(graph)$type2=g_mat[,5];
  return(graph)
}
plot_graph=function(graph,edgescale,arrowscale,thrs,color_list,layout){
  plot(graph, edge.lty=0, edge.arrow.size=0, layout=layout,
       vertex.shape='circle',vertex.size=8,
       vertex.color=ifelse(V(graph)$type=="rep","red","blue"),vertex.label.dist=2,
       vertex.label.cex=0.8)
  for (e in 1:ecount(graph)) {
    graph2 <- delete.edges(graph, E(graph)[(1:ecount(graph))[-e]])
    plot(graph2,edge.color='forestgreen',edge.lty=ifelse(E(graph)$type[e],1,2),
         edge.curved=(E(graph)$type)[e],
         edge.width=abs(E(graph)$weight[e])*edgescale,
         edge.arrow.size=abs(E(graph)$weight[e])*arrowscale, layout=layout, vertex.shape="none",
         vertex.label=NA, add=TRUE)
  }
}

load("twitter_layout.RData")
edgescale=5;arrowscale=1.5;

#variable importance network learned by the multinomial model
thrs=0.3
g_mat_MN=read.csv("../Political-tweets-data/twitter_MN_infl_graph.csv",header=F)
graph=mat_to_graph(g_mat_MN,name_list,party_list,thrs)
par(mar=c(0,0,0,0))
#i influences j
for (i in 1:2){
  for (j in 1:2){
    png(sprintf("fig/twitter_MN_varimportance_%s_%s.png",party_vec[i],party_vec[j]))
    graph_temp=delete.edges(graph, E(graph)[E(graph)$type1!=i|E(graph)$type2!=j])
    plot_graph(graph_temp, edgescale,arrowscale,thrs,color_list,l)
    dev.off()
  }
}


#variable importance network learned by the logistic-normal model
thrs=0.3;
g_mat_LN=read.csv("../Political-tweets-data/twitter_LN_infl_graph.csv",header=F)
graph=mat_to_graph(g_mat_LN,name_list,party_list,thrs)
par(mar=c(0,0,0,0))
#i influences j
for (i in 1:2){
  for (j in 1:2){
    png(sprintf("fig/twitter_LN_varimportance_%s_%s.png",party_vec[i],party_vec[j]))
    graph_temp=delete.edges(graph, E(graph)[E(graph)$type1!=i|E(graph)$type2!=j])
    plot_graph(graph_temp, edgescale,arrowscale,thrs,color_list,l)
    dev.off()
  }
}

#variable importance network learned by the mixture model
thrs=0.3
g_mat_mix=read.csv("../Political-tweets-data/twitter_mixed_infl_graph.csv",header=F)
graph=mat_to_graph(g_mat_mix,name_list,party_list,thrs);
par(mar=c(0,0,0,0))
for (i in 1:2){
  for (j in 1:2){
    png(sprintf("fig/twitter_mix_varimportance_%s_%s.png",party_vec[i],party_vec[j]))
    graph_temp=delete.edges(graph, E(graph)[E(graph)$type1!=i|E(graph)$type2!=j])
    plot_graph(graph_temp,edgescale,arrowscale,thrs,color_list,l)
    dev.off()
  }
}



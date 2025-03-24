####################################################
### Code patterned after: 
### Tutorial: Text Mining and NLP
### https://gatesboltonanalytics.com/?page_id=260
### Dr. Gates 
####################################################

library(ggplot2)
library(tm)
library(stringr)
library(wordcloud)
library(slam)
library(quanteda)
library(SnowballC)
library(arules)
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
library(gofastr)   ## for removing your own stopwords

library(textstem)  ## Needed for lemmatize_strings
library(amap)  ## for Kmeans
library(networkD3)


Tweetfile="0930_UkraineCombinedTweetsDeduped.csv"
setwd("C:/Machine_Learning/")

# Import the data and look at it
tweets <- read.csv(Tweetfile, nrows = 500)

head(tweets)

#remove rows with other languages than English

nrow(tweets)
tweets<-tweets[-which(tweets$language!="en"),]
## AFTER
nrow(tweets)


############################################
## Part 1: Cleaning the data
################################################

## LOOK AT Each Variable.
str(tweets)
(TweetVarNames<-names(tweets))
(TweetNumColumns<-ncol(tweets))

#plot retweet count vs. favorite count to look for correlations
ggplot(data = tweets, aes(x = retweetcount, y = favorite_count)) +
  geom_point() + ggtitle("favorites vs. retweets")

## mean and standard deviation of retweets and favorites
nrow(tweets)
summary(tweets$retweetcount)
mean(tweets$retweetcount, na.rm = T)
sd(tweets$retweetcount, na.rm = T)

summary(tweets$favorite_count)
mean(tweets$favorite_count, na.rm = T)
sd(tweets$favorite_count, na.rm = T)


#########################################################
##          Part 2: Text Mining and NLP             
#########################################################

##Load in the text corpus

text <- tweets$text
TweetCorpus <- Corpus(VectorSource(text))
TweetCorpus

(getTransformations())
(ndocs<-length(TweetCorpus))

##The following will show you that you read in all the documents
(summary(TweetCorpus))

# ignore extremely rare words i.e. terms that appear in less then 0.1% of the documents
(minTermFreq <- ndocs * 0.0001)

# ignore overly common words i.e. terms that appear in more than 50% of the documents
(maxTermFreq <- ndocs * 0.5)
(MyStopwords <- c("Ukraine", "war", "Russia"))

(STOPS <-stopwords('english'))


Tweets_dtm <- DocumentTermMatrix(TweetCorpus,
                                 control = list(
                                   stopwords = TRUE, 
                                   wordLengths=c(4, 10),
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE,
                                   #stemming = F,
                                   #stemWords=TRUE,
                                   remove_separators = TRUE,
                                   #stem=TRUE,
                                   stopwords("english"),
                                   bounds = list(global = c(minTermFreq, maxTermFreq))
                                 ))

## Look at word frequencies
(WordFreq <- colSums(as.matrix(Tweets_dtm)))
(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])
## Row Sums
(Row_Sum_Per_doc <- rowSums((as.matrix(Tweets_dtm))))

## Create a normalized version of Tweets_dtm
Tweets_M <- as.matrix(Tweets_dtm)
Tweets_M_N1 <- apply(Tweets_M, 1, function(i) i/sum(i))
## transpose
Tweets_Matrix_Norm <- t(Tweets_M_N1)
## Have a look at the original and the norm to make sure
(Tweets_M[c(1:3),c(1:8)])
(Tweets_Matrix_Norm[c(1:3),c(1:8)])

## Convert to matrix and view
Tweets_dtm_matrix = as.matrix(Tweets_dtm)
str(Tweets_dtm_matrix)
(Tweets_dtm_matrix[c(3:8),c(1:9)])

## Also convert to DF
Tweets_DF <- as.data.frame(as.matrix(Tweets_dtm))
str(Tweets_DF)
(Tweets_DF$said)
(nrow(Tweets_DF))
(ncol(Tweets_DF))

#build a word cloud
wordcloud(colnames(Tweets_dtm_matrix), Tweets_dtm_matrix[10, ], max.words = 100)
(head(sort(as.matrix(Tweets_dtm)[10,], decreasing = TRUE), n=20))



########### Frequencies and Associations ###################

## Find frequent words
(findFreqTerms(Tweets_dtm, 100))

## Find associations with a selected conf
(findAssocs(Tweets_dtm, 'war', 0.80))



############## Distance Measures ######################

m  <- Tweets_dtm_matrix
m_norm <- Tweets_dtm_matrix
distMatrix_E <- dist(m, method="euclidean")
print(distMatrix_E)
distMatrix_C <- dist(m, method="cosine")
print(distMatrix_C)
distMatrix_C_norm <- dist(m_norm, method="cosine")
print(distMatrix_C_norm)

############# Clustering #############################
## Hierarchical

## Euclidean
groups_E <- hclust(distMatrix_E,method="ward.D")
plot(groups_E, cex=0.9, hang=-1, main = "Euclidean")
rect.hclust(groups_E, k=4)

## From the NetworkD3 library
radialNetwork(as.radialNetwork(groups_E))
dendroNetwork(groups_E)

## Cosine Similarity
groups_C <- hclust(distMatrix_C,method="ward.D")
plot(groups_C, cex=0.9, hang=-1,main = "Cosine Sim")
rect.hclust(groups_C, k=4)

radialNetwork(as.radialNetwork(groups_C))
dendroNetwork(groups_C)

## Cosine Similarity for Normalized Matrix
groups_C_n <- hclust(distMatrix_C_norm,method="ward.D")
plot(groups_C_n, cex=0.9, hang=-1,main = "Cosine Sim and Normalized")
rect.hclust(groups_C_n, k=4)

radialNetwork(as.radialNetwork(groups_C_n))
dendroNetwork(groups_C_n)

### Per dr. Gates' notes: Cosine Sim works the best. Norm and not norm is about
## the same because the size of the novels are not sig diff.





####################   k means clustering -----------------------------
X <- m_norm

fviz_dist(distMatrix_C_norm, gradient = list(low = "#00AFBB", 
                                             mid = "white", high = "#FC4E07"))+
  ggtitle("Cosine Sim Normalized Distance Map")

#-

distance0 <- get_dist(m_norm,method = "euclidean")
fviz_dist(distance0, gradient = list(low = "#00AFBB", 
                                     mid = "white", high = "#FC4E07"))+
  ggtitle("Euclidean Distance Map")


#-
distance1 <- get_dist(m_norm,method = "manhattan")
fviz_dist(distance1, gradient = list(low = "#00AFBB", 
                                     mid = "white", high = "#FC4E07"))+
  ggtitle("Manhattan Distance Map")


#-
distance2 <- get_dist(m_norm,method = "pearson")
fviz_dist(distance2, gradient = list(low = "#00AFBB", 
                                     mid = "white", high = "#FC4E07"))+
  ggtitle("Pearson Distance Map")


#-
distance3 <- get_dist(m_norm,method = "canberra")
fviz_dist(distance3, gradient = list(low = "#00AFBB", 
                                     mid = "white", high = "#FC4E07"))+
  ggtitle("Canberra Distance Map")


#-
distance4 <- get_dist(m_norm,method = "spearman")
fviz_dist(distance4, gradient = list(low = "#00AFBB", 
                                     mid = "white", high = "#FC4E07"))+
  ggtitle("Spearman Distance Map")

Y <- t(X)
str(Y)

## k means
kmeansFIT_1 <- kmeans(Y, centers=5,  nstart=4)
(kmeansFIT_1)
(kmeansFIT_1$centers)
summary(kmeansFIT_1)
(kmeansFIT_1$cluster)
fviz_cluster(kmeansFIT_1, data = Y,main="Cluster Plot - Words")

## Run Kmeans methods
My_Kmeans1<-Kmeans(Y, centers=5,method = "euclidean")
fviz_cluster(My_Kmeans1, Y, main="Cluster Plot - Euclidean Words")

My_Kmeans1t<-Kmeans(X, centers=5,method = "euclidean")
fviz_cluster(My_Kmeans1t, X, main="Cluster Plot - Euclidean tweets")

My_Kmeans2<-Kmeans(Y, centers=5,method = "spearman")
fviz_cluster(My_Kmeans2, Y, main="Cluster Plot - Spearman words")

My_Kmeans2t<-Kmeans(X, centers=5,method = "spearman")
fviz_cluster(My_Kmeans2t, X, main="Cluster Plot - Spearman tweets")

My_Kmeans3<-Kmeans(Y, centers=5,method = "manhattan")
fviz_cluster(My_Kmeans3, Y, main="Cluster Plot - Manhattan words")

My_Kmeans3t<-Kmeans(X, centers=5,method = "manhattan")
fviz_cluster(My_Kmeans3t, X, main="Cluster Plot - Manhattan tweets")




############################# Elbow Methods ###################

fviz_nbclust(
  as.matrix(Tweets_dtm), 
  kmeans, 
  k.max = 9,
  method = "wss",
  diss = get_dist(as.matrix(Tweets_dtm), method = "manhattan")
)

fviz_nbclust(
  as.matrix(Tweets_dtm),
  kmeans, 
  k.max = 9,
  method = "wss",
  diss = get_dist(as.matrix(Tweets_dtm), method = "spearman")
)

fviz_nbclust(Tweets_DF, method = "silhouette", 
             FUN = hcut, k.max = 9)

#########################################################
##
## Patterned after https://rpubs.com/lgendrot/sentiment and 
## Dr. Gates' Tutorials on Naive Bayes, SVM nad Decision Trees 
## https://gatesboltonanalytics.com/?page_id=274
## https://gatesboltonanalytics.com/?page_id=280
## https://gatesboltonanalytics.com/?page_id=304
##
#########################################################

library(naivebayes)
library(e1071)
library("dplyr")
library("plyr")
library("class")
library(stopwords)
#install.packages("rpart.plot")
library(rpart)   ## FOR Decision Trees
library(rpart.plot)
library(rattle)  ## FOR Decision Tree Vis
#install.packages("randomForest")
library("randomForest")
#install.packages("ranger")
library("ranger")
library(arules)
library(dplyr)

TweetsFile="Tweets_Sentiment_Analysis_RoBERTa.csv"
setwd("C:/Machine_Learning/")

# Import the data and look at it
tweets <- read.csv(TweetsFile, nrows = 5000)
#tweets$Sentiment <- as.factor(tweets$Sentiment)
str(tweets)
head(tweets)
df0 = subset(tweets, select = -c(tweetid,hashtags) )
df <- subset(df0, df0$text != "" & df0$Sentiment != "")
df$Sentiment <- as.factor(df$Sentiment)

#######################################################################################
tokenize <- function(documents){
  # Lowercase all words for convenience
  doc <- tolower(documents)
  
  # Remove all #hashtags and @mentions
  doc <- gsub("(?:#|@)[a-zA-Z0-9_]+ ?", "", doc)
  
  # Remove words with more than 3 numbers in them (they overwhelm the corpus, and are uninformative)
  doc <- gsub("[a-zA-Z]*([0-9]{3,})[a-zA-Z0-9]* ?", "", doc)
  
  # Remove all punctuation
  doc <- gsub("[[:punct:]]", "", doc)
  
  # Remove all newline characters
  doc <- gsub("[\r\n]", "", doc)
  
  # Regex pattern for removing stop words
  stop_pattern <- paste0("\\b(", paste0(stopwords("en"), collapse="|"), ")\\b")
  doc <- gsub(stop_pattern, "", doc)
  
  # Replace whitespace longer than 1 space with a single space
  doc <- gsub(" {2,}", " ", doc)
  
  # Split on spaces and return list of character vectors
  doc_words <- strsplit(doc, " ")

  return(doc_words)
}
#######################################################################################






#######################################################################################
tfidf <- function(document, corpus){
  #Create a data frame out of a single document and its word frequency
  # For tweets this will be mostly 1's
  doc_f <- data.frame(unlist(table(document)))
  names(doc_f) <- c("Word", "Freq")
  
  #Get a data frame of the words in the corpus found in the current document
  in_doc <- intersect(doc_f$Word, corpus$Word)
  doc_f <- doc_f[doc_f$Word %in% in_doc, ]
  
  #Get a data frame of the words in the corpus not found in the current document
  #Set their frequency to 0
  not_in_doc <- data.frame(Word=setdiff(corpus$Word, document))
  not_in_doc$Freq <-0
  
  #Bind our two data frames, we now have frequencies for the words that are in our corpus, and 0s everywhere else
  tf <- rbind(doc_f, not_in_doc)
  tf$Word <- as.character(tf$Word)
  tf$Freq <- as.numeric(tf$Freq)
  
  #Order alphabetically again so it remains compatible with our corpus data frame
  tf <- tf[order(tf$Word), ]
  
  #Calculate the tfidf
  #log1p is the same as log(1+___)
  log_freq <- log1p(tf$Freq)
  log_doc_freq <- log1p(nrow(corpus)/corpus$n_docs)
  tf$tfidf <- log_freq * log_doc_freq
  
  #Divide by zero errors get NA values, but should be 0s
  tf$tfidf[is.na(tf$tfidf)] <- 0
  return(tf)
}
#######################################################################################







#######################################################################################
#Calculates accuracy
accuracy <- function(confusion_matrix){
  acc <- (confusion_matrix[1]+confusion_matrix[5]+confusion_matrix[9])/sum(confusion_matrix)
  return(accuracy=acc)
}
#######################################################################################





#Tokenize
tokens <- tokenize(df$text)
corpus_size=200

#get feature vectors
#rm(c)
all_words <- do.call(c, tokens)

#take the top words up to the length of corpus_size
#and reorder alphabetically
#This gives us an data frame of the most frequent words in our corpus, ordered alphabetically
#sized by the corpus_size parameter
corpusfreq <- data.frame(table(all_words))
names(corpusfreq) <- c("Word", "Freq")
corpusfreq$Word <- as.character(corpusfreq$Word)
corpusfreq$Freq <- as.numeric(corpusfreq$Freq)
corpusfreq <- corpusfreq[order(-corpusfreq$Freq), ]
corpusfreq <- corpusfreq[1:corpus_size, ]
corpusfreq <- corpusfreq[order(corpusfreq$Word), ]

# N docs is where we will store the document frequency (I.E how many documents a word appears in)
# We'll need this to calculate TF-IDF
corpusfreq$n_docs <- 0

# For every vector of words in our tokenized list, count how many times each word in our corpus occurs
for(token_list in tokens){
  if(length(token_list) > 0){
    t <- data.frame(table(token_list))
    names(t) <- c("Word", "n_docs")
    t$n_docs <- 1
    t_freq <- merge(x=corpusfreq, y=t, by="Word", all.x=TRUE)
    t_freq$n_docs.y[is.na(t_freq$n_docs.y)] <- 0
    corpusfreq$n_docs <- corpusfreq$n_docs + t_freq$n_docs.y    
  }
}
corpus <- corpusfreq


#Our feature matrix starts out as an all 0 matrix with N by C dimensions
feature_matrix <- matrix(0, length(tokens), nrow(corpus))


#For every document in our tokenized list, calculate the tfidf feature vector, and put it into our feature matrix row-wise
for(i in 1:length(tokens)){
  if(length(tokens[[i]])>0){
    feature_vector <- tfidf(tokens[[i]], corpus)$tfidf
    feature_matrix[i, 1:nrow(corpus)] <- feature_vector
  }
}


  
#The column names are the same as the alphabetical list of words in our corpus
#Unnecessary step, but useful for examining the resulting feature matrix
colnames(feature_matrix) <- corpus$Word
my_features <- data.frame(feature_matrix)
my_features_NL <- my_features


#Add the dependent variable for model fitting, I.E. the pre-labeled sentiment
my_features$Sentiment <- df$Sentiment
my_features$Sentiment <- as.factor(my_features$Sentiment)


##########################################################
##
##  Create the Testing and Training Sets         
##
########################################################


## This method works whether the data is in order or not.
X = 20   ## This will create a 1/20 split. 
## Of course, X can be any number.
every_X_index<-seq(1,nrow(my_features),X)

## Use these X indices to make the Testing and then
## Training sets:

test<-my_features[every_X_index, ]
train<-my_features[-every_X_index, ]

#write to csv
#write.csv(train, file="Tweets_train.csv", row.names = F)
#write.csv(test, file="Tweets_test.csv", row.names = F)

##################################### REMOVE AND SAVE LABELS...
## Test...--------------------------------
## Copy the Labels
test_Labels <- test$Sentiment

## Remove the labels
test_NL<-test[ , -which(names(test) %in% c("Sentiment"))]

## Train...--------------------------------
## Copy the Labels
train_Labels <- train$Sentiment

## Remove the labels
train_NL<-train[ , -which(names(train) %in% c("Sentiment"))]



#Formula for each model
form <- as.formula(paste("Sentiment~", paste(setdiff(names(test), c("Sentiment")), collapse="+")))





#################################################
##
## decision tree
## 
#################################################


# discretize 
train_discretized<-train_NL[ , -which(names(train_NL) %in% c("V1"))]
train_discretized <- train_discretized %>% mutate_if(is.numeric, funs(discretize(., method="fixed", 
breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 1000))))
train_discretized$Sentiment <- train_Labels

#use GINI
DT3 <- rpart(train_discretized$Sentiment ~ ., data = train_discretized, method="class")
summary(DT3)
plotcp(DT3) ## This is the cp plot
rattle::fancyRpartPlot(DT3,main="Decision Tree: discretized features, GINI", cex=.5)

test_NL_discretized<-test_NL[ , -which(names(test_NL) %in% c("V1"))]
test_NL_discretized <- test_NL_discretized %>% mutate_if(is.numeric, funs(discretize(., method="fixed", 
breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 1000))))

DT3_Prediction= predict(DT3, test_NL_discretized, type="class")
## Confusion Matrix
table(DT3_Prediction,test_Labels)
accuracy(table(DT3_Prediction,test_Labels))

#use GINI
DT <- rpart(train$Sentiment ~ ., data = train, cp=0.003, method="class")
#summary(DT)
plotcp(DT) ## This is the cp plot
rattle::fancyRpartPlot(DT,main="Decision Tree: cp=0.003, GINI", cex=.5)
DT_Prediction= predict(DT, test_NL, type="class")
## Confusion Matrix
table(DT_Prediction,test_Labels)
accuracy(table(DT_Prediction,test_Labels))


#use information
DT2 <- rpart(train$Sentiment ~ ., data = train, cp=0.003, method="class",
             parms = list(split="information"),minsplit=2)
#summary(DT2)
plotcp(DT2) ## This is the cp plot
rattle::fancyRpartPlot(DT2,main="Decision Tree: cp=0.003, information", cex=.5)
DT2_Prediction= predict(DT2, test_NL, type="class")
## Confusion Matrix
table(DT2_Prediction,test_Labels)
accuracy(table(DT2_Prediction,test_Labels))




#Random forest
m_randomforest <- ranger(dependent.variable.name="Sentiment", data=train, write.forest=TRUE)
pred_randomforest <- predict(m_randomforest, test_NL)
table(pred_randomforest$predictions, test_Labels)
accuracy(table(pred_randomforest$predictions, test_Labels))




#################################################
##
## SVM
## 
#################################################


#tune the SVM by altering the cost
#tuned_cost <- svm(form, data=train, kernel = "linear", ranges=list(cost = c(0.01,0.1,1,10,100)), type="C")
#summary(tuned_cost)

#tune the SVM by altering the cost
#tuned_cost <- svm(form, data=train, kernel = "polynomial", ranges=list(cost = c(0.01,0.1,1,10,100)), type="C")
#summary(tuned_cost)

#tune the SVM by altering the cost
#tuned_cost <- svm(form, data=train, kernel = "radial", ranges=list(cost = c(0.01,0.1,1,10,100)), type="C")
#summary(tuned_cost)

#Support vector machine with different kernels
m_svm <- svm(form, data=train, kernel = "linear", cost = 1, type="C")
print(m_svm)
pred_svm <- predict(m_svm, test_NL)
table(pred_svm, test_Labels)
accuracy(table(pred_svm, test_Labels))
plot(m_svm, data=train, russia~ukraine)

m_svm <- svm(form, data=train, kernel = "polynomial", cost = 1, type="C")
print(m_svm)
pred_svm <- predict(m_svm, test_NL)
table(pred_svm, test_Labels)
accuracy(table(pred_svm, test_Labels))
plot(m_svm, data=train, russia~ukraine)

m_svm <- svm(form, data=train, kernel = "radial", cost = 1, type="C")
print(m_svm)
pred_svm <- predict(m_svm, test_NL)
table(pred_svm, test_Labels)
accuracy(table(pred_svm, test_Labels))
plot(m_svm, data=train, russia~ukraine)



#################################################
##
## NAIVE BAYES
## 
#################################################



#Naive Bayes algorithm
NB_e1071_2<-naiveBayes(x=train_NL, y=train_Labels,laplace=1)
NB_e1071_Pred <- predict(NB_e1071_2, test_NL,laplace=1)
table(NB_e1071_Pred,test_Labels)
accuracy(table(NB_e1071_Pred,test_Labels))




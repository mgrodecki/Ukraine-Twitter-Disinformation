####################################################
### Code patterned after the ARM with Twitter lecture: 
### https://gatesboltonanalytics.com/wp-content/uploads/2022/08/ARM-_with-Twitter_-Lecture-2021-Gates_Updated-2022.pptx
### Dr. Gates 
####################################################
library(viridis)
library(arules)
library(TSP)
library(data.table)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)
library(stylo)
library(stringr)
library(readr)

TweetsFile="1124_UkraineCombinedTweetsDeduped.csv"
TransactionTweetsFile="Transactions_1124_UkraineCombinedTweetsDeduped.csv"
CleanedTransactionTweetsFile="Cleaned_Transactions_1124_UkraineCombinedTweetsDeduped.csv"
setwd("C:/Machine_Learning/")

Trans <- file(TransactionTweetsFile)

# Import the data and look at it
tweets <- read.csv(TweetsFile, nrows = 5000)
head(tweets)

#remove rows with other languages than english
nrow(tweets)
tweets<-tweets[-which(tweets$language!="en"),]
## AFTER
head(tweets)

#tokenize to words
Tokens <- tokenizers::tokenize_words(tweets$text[1],stopwords=stopwords::stopwords("en"),
          lowercase=TRUE,strip_punct=TRUE,strip_numeric=TRUE,simplify=TRUE)

#write squished tokens
cat(unlist(str_squish(Tokens)),"\n",file=Trans,sep=",")
close(Trans)

#append remaining lists of tokens
Trans <- file(TransactionTweetsFile,open="a")
for(i in 2:nrow(tweets)){
  Tokens <- tokenizers::tokenize_words(tweets$text[i],stopwords=stopwords::stopwords("en"),
                                       lowercase=TRUE,strip_punct=TRUE,strip_numeric=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),"\n",file=Trans,sep=",")
}
close(Trans)

#detect <- str_detect(Tokens, "[^\\w\\d\\s,]")

#detach(package:tm, unload=TRUE)
#library(arules)
#read in the tweet transactions
TweetTrans <- read.transactions(TransactionTweetsFile,format="basket",sep=",")
inspect(TweetTrans)

#see the most frequent words
Sample_Trans <- sample(TweetTrans, 20)
summary(Sample_Trans)


#read the transactions into a dataframe
TweetDF <- read.csv(TransactionTweetsFile, header=FALSE,sep=",")
head(TweetDF)

#convert all cols to char
TweetDF <- TweetDF %>%
  mutate_all(as.character)
str((TweetDF))

#remove unnecessary words
TweetDF[TweetDF=="t.co"]<-""
TweetDF[TweetDF=="rt"]<-""
TweetDF[TweetDF=="http"]<-""
TweetDF[TweetDF=="https"]<-""

#clean with grepl
MyDF<-NULL
MyDF2<-NULL
for(i in 1:ncol(TweetDF)){
  MyList=c()
  MyList=c(MyList,grepl("[[:digit:]]",TweetDF[[i]]))
  
  MyList2=c()
  MyList2=c(MyList2,grepl("[[[:digit:]]]A-z]{14,}",TweetDF[[i]]))

  MyDF<-cbind(MyDF,MyList)
  
  MyD2<-cbind(MyDF2,MyList2)
}
TweetDF[MyDF]<-""
TweetDF[MyDF2]<-""
(head(TweetDF,10))

#write cleaned file
write_csv(TweetDF, CleanedTransactionTweetsFile)

#read in the tweet transactions
TweetTrans2 <- read.transactions(CleanedTransactionTweetsFile,format="basket",sep=",",skip=1)
inspect(TweetTrans2)


##### Use apriori to get the RULES
Rules = arules::apriori(TweetTrans2, parameter = list(support=.03, confidence=.5, minlen=2, maxlen=4))
(summary(Rules))

SortedRules <- sort(Rules, by="lift", decreasing=TRUE)
subrules <- head(sort(SortedRules, by="coverage", decreasing=TRUE),10)
inspect(subrules)
plot(subrules, method="graph", engine="htmlwidget")


## Sort rules by a measure such as conf, sup, or lift
SortedRules2 <- sort(Rules, by="confidence", decreasing=TRUE)
subrules2 <- head(sort(SortedRules2, by="coverage", decreasing=TRUE),10)
inspect(subrules2)
plot(subrules2, method="graph", engine="htmlwidget")



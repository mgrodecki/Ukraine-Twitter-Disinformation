##################################
## The Twitter API v2 in R
##################################
#Sys.setenv(BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADJFhAEAAAAAiBUou%2BuZbQOcTUuuqkKjyhzyZhs%3DqdY34O1dbNrMD0lj4d1NkqgEaF9a0gYGV3ffRfb6k5DeUj9gE3")

#install.packages("httr")
#install.packages("jsonlite")
#install.packages("dplyr")

packages <- c("httr", "jsonlite", "dplyr")
### checking if packages are already installed and installing if not
for(i in packages){
  if(!(i %in% installed.packages()[, "Package"])){
    install.packages(i)
  }
  library(i, character.only = TRUE) ## load packages
}

require(httr)
require(jsonlite)
require(dplyr)

bearer_token <- Sys.getenv("$BEARER_TOKEN")
headers <- c(`Authorization` = sprintf('Bearer %s', bearer_token))

params <- list(`user.fields` = 'description', `expansions` = 'pinned_tweet_id')

handle <- readline('$USERNAME')
url_handle <- sprintf('https://api.twitter.com/2/users/by?usernames=%s', handle)

response <- httr::GET(url = url_handle, httr::add_headers(.headers = headers), query = params)
obj <- httr::content(response, as = "text")
print(obj)

json_data <- fromJSON(obj, flatten = TRUE) %>% as.data.frame
View(json_data)

final <-
  sprintf(
    "Handle: %s\nBio: %s\nPinned Tweet: %s",
    json_data$data.username,
    json_data$data.description,
    json_data$includes.tweets.text
  )

cat(final)



##-------------------------------------------------------------
#############################################################
### The Twitter API   ########################################
############################################################
## You MUST first apply for and get a Twitter Dev Account
## Create a new App on the account AND
## get the access codes
## https://developer.twitter.com/en/apply-for-access.html
##############################################################

#install.packages("twitteR")
#install.packages("ROAuth")
#install.packages("rtweet")
#install.packages("base64enc")
packages <- c("ROAuth", "rtweet", "openssl", "httpuv", "twitteR")
### checking if packages are already installed and installing if not
for(i in packages){
  if(!(i %in% installed.packages()[, "Package"])){
    install.packages(i)
  }
  library(i, character.only = TRUE) ## load packages
}

#library(twitteR)
#library(rtweet)
#library(ROAuth)
library(jsonlite)
#library(base64enc)
#library(openssl)

setwd("C:/Machine_Learning/")
## The above is where my Twitter Dev passcodes are located.
## This will NOT BE THE SAME FOR YOU

## What is going on here?
## Here - I have placed the 4 Twitter passcodes into a .txt file. When I need them, I read them out
## of the file. This is better and safer than coding them in directly. 
filename=filename="C:/Machine_Learning/TwitterConKey_ConSec_AccTok_AccSec.txt"
(tokens<-read.csv(filename, header=TRUE, sep=","))

## This is important. You need to assure that your codes read in as character strings.
(consumerKey=as.character(tokens$consumerKey))  ## tokens is what I named my passcodes read from my file.
consumerSecret=as.character(tokens$consumerSecret)
access_Token=as.character(tokens$access_Token)
access_Secret=as.character(tokens$access_Secret)

## Do not remove this
requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

## Set up  - log in - to Twitter
setup_twitter_oauth(consumerKey,consumerSecret,access_Token,access_Secret)
## Use the twitteR library, method searchTwitter to search a specific hashtag, number of tweets
## I am getting three here, and the date (optional). 
## DO NOT get too many Tweets at once while you are testing your code or YOU WILL RUN OUT FOR THE DAY
## Twitter LIMITS the number of Tweets you can grab per day and at once.
Search1<-twitteR::searchTwitter("#Gators",n=3, since="2020-03-01")
(Search_DF2 <- twListToDF(Search1))

(Search_DF2$text[1])


########## Place Tweets in a new file ###################
FName = "MyFileExample.txt"
## Start the file
MyFile <- file(FName)
## Write Tweets to file
cat(unlist(Search_DF2), " ", file=MyFile, sep="\n")
close(MyFile)
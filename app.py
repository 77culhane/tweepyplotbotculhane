#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import tweepy
import time

# API keys
import api_keys

# Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Incorporate matplotlib inline
# get_ipython().magic(u'matplotlib inline')

# Twitter API Keys
consumer_key = api_keys.consumer_key
consumer_secret = api_keys.consumer_secret
access_token = api_keys.access_token
access_token_secret = api_keys.access_token_secret
user = api_keys.user


# In[2]:


# Function for Analyzing Tweets
def tweetbot():

    #prepare api keys for call 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    jsonreader = tweepy.parsers.JSONParser()
    api = tweepy.API(auth, parser=jsonreader)

    # ## Grab Most Recent Mention

    # run a search for the command phrase "@{user} Analyze: @{target_account}"
    analysis_request = api.search(q=f"{user} Analyze:")
    #parse the tweet for 
    command = analysis_request["statuses"][0]["text"]
    requesting_user = analysis_request["statuses"][0]["user"]["screen_name"]

    # Input Tweet
    words = []

    # Split it to determine the target account
    try:
        words = command.split("Analyze:")
        #parse the username who made the request
        target_account = f"{words[1].strip()}"

        #confirm target_account
        print("Target Account: " + target_account)
        print("Requesting User: " + requesting_user)

    except Exception:
        raise

    #check timeline for duplicates of target_account
    mytweets = api.user_timeline()

    #confirm target_account has never been tweeted before
    repeat = False

    #for tweet in mytweets:
    #    if target_account in tweet["text"]:
    #        repeat = True
    #        print("Sorry. Repeat detected!")

    #    else:
    #        continue


    #if the target_account is not a duplicate
    if not (repeat):

        # make empty dictionary into which the data will be appended
        tweets_dict = {
            "tweet_source": [],
            "tweet_text": [],
            "tweet_date": [],
            "tweet_vader_score": [],
            "tweet_neg_score": [],
            "tweet_pos_score": [],
            "tweet_neu_score": []
        }

        #loop through 25 pages for each News Source (where x = page number);
        #at 20 tweets per page, this will yield 500 tweets
        for x in range(25):

            #run the API call
            target_tweets = api.user_timeline(target_account, page=x)

            #append each tweet to tweets_dict
            for tweet in target_tweets:
                tweets_dict["tweet_source"].append(tweet["user"]["name"])
                tweets_dict["tweet_text"].append(tweet["text"])
                tweets_dict["tweet_date"].append(tweet["created_at"])
                #set up vader variable
                runvader = analyzer.polarity_scores(tweet["text"])
                #run vader to produce sentiment values
                tweets_dict["tweet_vader_score"].append(runvader["compound"])
                tweets_dict["tweet_pos_score"].append(runvader["pos"])
                tweets_dict["tweet_neu_score"].append(runvader["neu"])
                tweets_dict["tweet_neg_score"].append(runvader["neg"])

        #convert to dataframe
        df = pd.DataFrame(tweets_dict, columns=["tweet_source",
                                                     "tweet_text",
                                                     "tweet_date",
                                                     "tweet_vader_score",
                                                     "tweet_pos_score",
                                                     "tweet_neu_score",
                                                     "tweet_neg_score"])


    if not (repeat):

        # Convert dates (currently strings) into datetimes
        df["tweet_date"] = pd.to_datetime(df["tweet_date"])
        
        # Sort the dataframe by date
        df.sort_values("tweet_date", inplace=True)
        df.reset_index(drop=True, inplace=True)


    if not (repeat):

        # Clear Plot
        plt.clf()

        # Build scatter plot for tracking tweet polarity by tweet history
        # Note how a few data munging tricks were used to obtain (-100 -> 0
        # ticks)
        plt.plot(np.arange(-len(df["tweet_vader_score"]), 0, 1),
                 df["tweet_vader_score"], marker="o", linewidth=0.5,
                 alpha=0.8, label="%s" % target_account)

        # Incorporate the other graph properties
        plt.title("Sentiment Analysis of Tweets (%s)" % time.strftime("%x"))
        plt.ylabel("Tweet Polarity")
        plt.xlabel("Tweets Ago")
        plt.xlim([-len(df["tweet_vader_score"]) - 7, 7])
        plt.ylim([-1.05, 1.05])
        plt.grid(True)

        # Create a legend
        lgnd = plt.legend(fontsize="small", mode="Expanded",
                          numpoints=1, scatterpoints=1,
                          loc="upper left", bbox_to_anchor=(1, 1),
                          title="Tweets", labelspacing=0.5)

        # Save the figure (and account for the legend being outside the plot)
        file_path = "analysis/" + target_account + ".png"
        plt.savefig(file_path, bbox_extra_artists=(lgnd, ),
                    bbox_inches='tight')

    # ## Tweet Plot

    if not (repeat):

        # Tweet out the image and mention the user who requested it
        api.update_with_media(file_path,
                              f"(machine version) New Tweet Analysis: {target_account} (Thx @{requesting_user}!!)")


# In[ ]:


# Run the Analyze Tweets Function Every 5 minutes
while(True):
    tweetbot()
    print("finished")
    time.sleep(300)


# In[ ]:





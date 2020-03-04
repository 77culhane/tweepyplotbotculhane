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

# Function for Analyzing Tweets
def AnalyzeTweets():

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

    # Specified Twitter Account
    target_account = ""

    # Grab the most recent command tweet
    

    # Split it to determine the target account
    try:
        words = command.split("Analyze:")
        target_account = words[1].strip()

        # Confirm target_account
        print("Target Account: " + target_account)
        print("Requesting User: " + requesting_user)

    except Exception:
        raise

    # ## Confirm Non-Repeat

    # Grab Self Tweets
    tweets = api.user_timeline()

    # Confirm the target account has never been tweeted before
    repeat = False

    #for tweet in tweets:
    #    if target_account in tweet["text"]:
    #        repeat = True
    #        print("Sorry. Repeat detected!")

    #    else:
    #        continue

    # ## Run Sentiment Analysis

    # If the tweet specifies a unique account run the analysis
    if not (repeat):

        # Create a generic dictionary for holding all tweet information
        tweet_data = {
            "tweet_source": [],
            "tweet_text": [],
            "tweet_date": [],
            "tweet_vader_score": [],
            "tweet_neg_score": [],
            "tweet_pos_score": [],
            "tweet_neu_score": []
        }

        # Grab 500 tweets from the target source
        for x in range(25):

            # Grab the tweets
            tweets = api.user_timeline(target_account, page=x)

            # For each tweet store it into the dictionary
            for tweet in tweets:

                # All data is grabbed from the JSON returned by Twitter
                tweet_data["tweet_source"].append(tweet["user"]["name"])
                tweet_data["tweet_text"].append(tweet["text"])
                tweet_data["tweet_date"].append(tweet["created_at"])

                # Run sentiment analysis on each tweet using Vader
                tweet_data["tweet_vader_score"].append(
                    analyzer.polarity_scores(tweet["text"])["compound"])
                tweet_data["tweet_pos_score"].append(analyzer.polarity_scores(
                    tweet["text"])["pos"])
                tweet_data["tweet_neu_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neu"])
                tweet_data["tweet_neg_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neg"])

    if not (repeat):

        # Create a generic dictionary for holding all tweet information
        tweet_data = {
            "tweet_source": [],
            "tweet_text": [],
            "tweet_date": [],
            "tweet_vader_score": [],
            "tweet_neg_score": [],
            "tweet_pos_score": [],
            "tweet_neu_score": []
        }

        # Grab 500 tweets from the target source
        for x in range(25):

            # Grab the tweets
            tweets = api.user_timeline(target_account, page=x)

            # For each tweet store it into the dictionary
            for tweet in tweets:

                # All data is grabbed from the JSON returned by Twitter
                tweet_data["tweet_source"].append(tweet["user"]["name"])
                tweet_data["tweet_text"].append(tweet["text"])
                tweet_data["tweet_date"].append(tweet["created_at"])

                # Run sentiment analysis on each tweet using Vader
                tweet_data["tweet_vader_score"].append(
                    analyzer.polarity_scores(tweet["text"])["compound"])
                tweet_data["tweet_pos_score"].append(analyzer.polarity_scores(
                    tweet["text"])["pos"])
                tweet_data["tweet_neu_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neu"])
                tweet_data["tweet_neg_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neg"])

        # Store the final contents into a DataFrame
        tweet_df = pd.DataFrame(tweet_data, columns=["tweet_source",
                                                     "tweet_text",
                                                     "tweet_date",
                                                     "tweet_vader_score",
                                                     "tweet_pos_score",
                                                     "tweet_neu_score",
                                                     "tweet_neg_score"])

        # Visualize the DataFrame
        tweet_df.head()

    if not (repeat):

        # Convert dates (currently strings) into datetimes
        tweet_df["tweet_date"] = pd.to_datetime(tweet_df["tweet_date"])

        # Sort the dataframe by date
        tweet_df.sort_values("tweet_date", inplace=True)
        tweet_df.reset_index(drop=True, inplace=True)

        # Preview the data to confirm data is sorted
        tweet_df.head()

    if not (repeat):

        # Clear Plot
        plt.clf()

        # Build scatter plot for tracking tweet polarity by tweet history
        # Note how a few data munging tricks were used to obtain (-100 -> 0
        # ticks)
        plt.plot(np.arange(-len(tweet_df["tweet_vader_score"]), 0, 1),
                 tweet_df["tweet_vader_score"], marker="o", linewidth=0.5,
                 alpha=0.8, label="%s" % target_account)

        # Incorporate the other graph properties
        plt.title("Sentiment Analysis of Tweets (%s)" % time.strftime("%x"))
        plt.ylabel("Tweet Polarity")
        plt.xlabel("Tweets Ago")
        plt.xlim([-len(tweet_df["tweet_vader_score"]) - 7, 7])
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
                              f"New Tweet Analysis: {target_account} (Thx @{requesting_user}!!)")

# Run the Analyze Tweets Function Every 5 minutes
while(True):
    AnalyzeTweets()
    print("finished")
    time.sleep(30)
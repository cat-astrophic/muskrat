# This script uses snscrape to collect relevant tweets

# Importing required modules

import snscrape.modules.twitter as sntwitter
import pandas as pd

# Where to write raw data

writeto = 'F:/muskrat/data/snscrape.csv'

# Creating list to append tweet data to

tweets_list = []

# Using TwitterSearchScraper to scrape data and append tweets to list

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('elon OR musk OR twitter since:2022-10-01 until:2022-12-01').get_items()):

    tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above

tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_df.to_csv(writeto, index = False)


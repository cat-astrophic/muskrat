# This script performs sentiment analysis on the harvested tweets

# Importing required modules

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Where to write the sentiment analysis outputs

writeto = 'D:/muskrat/data/'

# Read in the tweets

td = pd.read_csv(writeto + 'raw_data.csv')

# Sentiment analysis

# Tokenizing the twitter data and removing stopwords

tweets = [str(tweet) for tweet in td.Text]
stop_words = set(stopwords.words('english'))
clean_tweets = []

for tweet in tweets:
    
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    
    for w in word_tokens:
        
        if w not in stop_words:
            
            filtered_sentence.append(w)
    
    clean_tweets.append(filtered_sentence)

# Lemmatizing

lemon = WordNetLemmatizer()

for t in range(len(clean_tweets)):
    
    res = []
    
    for w in clean_tweets[t]:
        
        res.append(lemon.lemmatize(w))
    
    clean_tweets[t] = res

# Stemming

ps = PorterStemmer()

for t in range(len(clean_tweets)):
    
    res = []
    
    for w in clean_tweets[t]:
        
        res.append(ps.stem(w))
        
    clean_tweets[t] = res

# Remove usernames from @s

for t in range(len(clean_tweets)):
    
    ws = clean_tweets[t]
    uns = [ws[w] for w in range(1,len(ws)) if ws[w-1] == '@']
    clean_tweets[t] = [c for c in clean_tweets[t] if c not in uns]

# Remove symbols

baddies = ['@', '#', '$', '%', '&', '*', ':', ';', '"', '.', ',', '/', '!',
           "'s", 'http', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

for t in range(len(clean_tweets)):
    
    clean_tweets[t] = [c for c in clean_tweets[t] if c not in baddies]

# The data is now prepped and ready to get all sentimental

sad = SentimentIntensityAnalyzer()

tweets = [' '.join(t) for t in clean_tweets]
svals = []
emotes = []

for t in tweets:
    
    svals.append(sad.polarity_scores(t))
    emotes.append(NRCLex(t).affect_frequencies)

# Creating lists of values for each category in svals

neg = []
neu = []
pos = []
comp = []

for s in svals:
    
    neg.append(s['neg'])
    neu.append(s['neu'])
    pos.append(s['pos'])
    comp.append(s['compound'])

# Adding the sentiment analysis scores to the main dataframe

neg = pd.Series(neg, name = 'Negative')
neu = pd.Series(neu, name = 'Neutral')
pos = pd.Series(pos, name = 'Positive')
comp = pd.Series(comp, name = 'Compound')

td = td.reset_index(drop = True)
td = pd.concat([td, neg, neu, pos, comp], axis = 1)

# Parsing and adding emotions data

anger = []
disgust = []
negative = []
joy = []
positive = []
anticipation = []
fear = []
sadness = []
trust = []
surprise = []

for e in emotes:
    
    anger.append(max(0,e['anger']))
    disgust.append(max(0,e['disgust']))
    negative.append(max(0,e['negative']))
    joy.append(max(0,e['joy']))
    positive.append(max(0,e['positive']))
    anticipation.append(max(0,e['anticip']))
    fear.append(max(0,e['fear']))
    sadness.append(max(0,e['sadness']))
    trust.append(max(0,e['trust']))
    surprise.append(max(0,e['surprise']))

# Adding the sentiment analysis scores to the main dataframe

anger = pd.Series(anger, name = 'Anger')
disgust = pd.Series(disgust, name = 'Disgust')
negative = pd.Series(negative, name = 'Negative_E')
joy = pd.Series(joy, name = 'Joy')
positive = pd.Series(positive, name = 'Positive_E')
anticipation = pd.Series(anticipation, name = 'Anticipation')
fear = pd.Series(fear, name = 'Fear')
sadness = pd.Series(sadness, name = 'Sadness')
trust = pd.Series(trust, name = 'Trust')
surprise = pd.Series(surprise, name = 'Surprise')

td = pd.concat([td, anger, disgust, negative, joy, positive, anticipation, fear, sadness, trust, surprise], axis = 1)

# Writing td to file

td.to_csv(writeto + 'sentiment_data.csv', index = False)


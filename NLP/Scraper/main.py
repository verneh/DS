# import

import tweepy
from tweepy import OAuthHandler
import pandas as pd

print("This works!")

# access credentials goes here.

access_token = '1314488361059459074-mHc1y8qvn2nH8w0moBDOT3H4tYLuGv'
access_token_secret = 'pIdilkEr0x4dv1PZZsKjTExuRpJHfRTNiBnUqajK9qUMR'
consumer_key = 'vjv2lgGRaDWpq2xHCMG8nrfKe'
consumer_secret = 'KC3FgFCUSWSlFdeT8U8AiVIaeL2QtFdw9LX9rm9d3SHioVpiBX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# tweets

tweets = []

count = 1

# anything that includes BBCWorld.
for tweet in tweepy.Cursor(api.search, q="Danny Green", count=450, since='2020-10-9').items(1000):

    print(count)
    count += 1

    try:
        data = [tweet.created_at, tweet.id, tweet.text, tweet.user._json['screen_name'], tweet.user._json['name'], tweet.user._json['created_at'], tweet.entities['urls'],
                tweet.user._json['location'], tweet.user._json['description']]
        data = tuple(data)
        tweets.append(data)

    except tweepy.TweepError as e:
        print(e.reason)
        continue

    except StopIteration:
        break


df = pd.DataFrame(tweets, columns = ['created_at','tweet_id', 'tweet_text', 'screen_name', 'name', 'account_creation_date', 'urls', 'location', 'description'])

df.to_csv('tweets.csv', index=False)
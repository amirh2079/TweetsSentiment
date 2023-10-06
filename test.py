from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob



def clean_text(unserem_text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", unserem_text).split())
    
    
    
    
df = pd.read_csv("E:/programming/Twitter - Copy/neu/Tweets.csv")
df.index = df.tweet_id



cat = {'negative':-1,'neutral':0, 'positive':1}
df['airline_sentiment'] = df['airline_sentiment'].map(cat)

# y = df.airline_sentiment



text = df.text
list_text = text.to_list()
list_sentiment =[]
average_pol_sub = []
tweet_sentiment = []

for i in range(0,len(list_text)):
    list_text[i] = clean_text(list_text[i])
    ana = TextBlob(list_text[i])
    
    if len(list_text[i]) < 7 and ana.sentiment.polarity > 0:
        tweet_sentiment.append(1)
    
    elif len(list_text[i]) < 7 and ana.sentiment.polarity < 0:
        tweet_sentiment.append(-1)
    
    elif len(list_text[i]) >= 7 and ana.sentiment.polarity >= 0.3:
        tweet_sentiment.append(1)
    elif ana.sentiment.polarity >=0 and ana.sentiment.polarity <0.3:
        tweet_sentiment.append(0)
    else:
        tweet_sentiment.append(-1)

    
#print(tweet_sentiment)
#print(y.head(10))


mein_df = pd.DataFrame(tweet_sentiment, columns=['mein_sentiment'], index = [df.index])
mein_df['airline_sentiment'] = (df.airline_sentiment).to_list()
mein_df['tweet text'] = list_text

mein_df['comparison_column'] = np.where(mein_df["mein_sentiment"] == mein_df["airline_sentiment"], 1, 0)

# time_mein_sentiment = pd.Series(data=mein_df['mein_sentiment'].values)
# time_mein_sentiment.plot(figsize=(16, 4), label="mein_sentiment", legend=True)

# time_airline_sentiment = pd.Series(data=mein_df['airline_sentiment'].values)
# time_airline_sentiment.plot(figsize=(16, 4), label="airline_sentiment", legend=True)

# plt.show()

#############


list_airline_sentiment_example = []
list_mein_sentiment_example = []
for i in range(600, 800):
    list_mein_sentiment_example.append(tweet_sentiment[i])
    list_airline_sentiment_example.append((((df.airline_sentiment).to_list())[i]))
                
example_df = pd.DataFrame(list_mein_sentiment_example, columns=['mein_sentiment_example'])
example_df['airline_sentiment_example'] = list_airline_sentiment_example

time_mein_sentiment_example = pd.Series(data=example_df['mein_sentiment_example'].values)
time_mein_sentiment_example.plot(figsize=(16, 4), label="mein_sentiment_example", legend=True)

time_airline_sentiment_example = pd.Series(data=example_df['airline_sentiment_example'].values)
time_airline_sentiment_example.plot(figsize=(16, 4), label="airline_sentiment_example", legend=True)
plt.show()
##############

mein_df.to_excel("mein_df.xlsx")
print('done')

# cols = ['tweet_id', 'negativereason', 'negativereason_confidence',
#        'airline_sentiment_gold', 'name', 'negativereason_gold',
#        'retweet_count', 'tweet_coord', 'tweet_created',
#        'tweet_location', 'user_timezone', 'text']
# df.drop(cols, axis = 1, inplace=True)

# df = df.fillna(df.mean())
# x = df

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

# # create tree model
# model1 = DecisionTreeClassifier(max_depth=2)
# model1.fit(x_train, y_train)
# y_pred = model1.predict(x_test)
# acc_score1 = accuracy_score(y_test, y_pred)
# #print('Decision Tree:', acc_score1)

# #print(list_text)


# #print(df.columns)
# #ana = TextBlob("it was not a good night") 
# #print(ana.polarity)


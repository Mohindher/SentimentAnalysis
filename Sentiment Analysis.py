"""
Perfomring Sentiment Analysis

@author: Mohindher.Thirumalai
"""

from textblob import TextBlob
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


"""----------------------------Sentiment Analysis on Feedbacks------------------------------------------"""

"""
Polarity: It measures the sentiment of the text on a scale from -1 to 1.
            Where -1 indicates a negative sentiment, 1 indicates a positive sentiment, and 0 indicates a neutral sentiment. 
            In the context of sentiment analysis, the polarity value tells us how positive, negative, or neutral the text is.

Subjectivity: It measures the subjectivity of the text on a scale from 0 to 1.
            where 0 indicates an objective text (factual, unbiased), 
            1 indicates a highly subjective text (opinionated, personal), 
            and values in between indicate varying degrees of subjectivity.
"""

feedbacks = ['Course Work is awesome.', 'Course Work was bad.', 'Course Work was Ok.']

for feedback in feedbacks:
    b = TextBlob(feedback)
    print(f"Feedback: {feedback}")
    print(f"Sentiment: {b.sentiment}\n")




"""----------------------------Sentiment Analysis on Text Data------------------------------------------"""

"""
Keep in mind that sentiment analysis at the word level may not always accurately capture the overall sentiment of a sentence or text. 
TextBlob's sentiment analysis is based on a pre-trained model and may not work perfectly for all words or contexts.
"""

# Load and preprocess the text data
dataset = pd.read_csv('C:/SentimentAnalysis.txt')
dataset = dataset.to_string(index=False)
dataset = re.sub("[^A-Za-z0-9]+", " ", dataset)

# Tokenization
Tokens = word_tokenize(dataset)
print("Tokenization Result:\n", Tokens)

# Frequency distribution of tokens
fdist = FreqDist()
for word in Tokens:
    fdist[word.lower()] += 1

# Plotting the most common words
print("\nMost Common Words:\n", fdist.most_common(20))

# Stemming
"""
Stemming is a text normalization technique to reduce words to their base or root form. 
The purpose of stemming is to convert different forms of a word to a common base form. 
Stemming is particularly useful for reducing the vocabulary size and improving the efficiency of text processing tasks.
In stemming, words are reduced to their root form by removing prefixes or suffixes. For example:

"running" becomes "run"
"jumps" becomes "jump"
"playing" becomes "play"
"""
pst = PorterStemmer()
stemmed_tokens = [pst.stem(word) for word in Tokens]

# Removing stop words such as and,the,is,in,of,on,etc..
stop_words = stopwords.words('english')
filtered_tokens = [word for word in stemmed_tokens if word not in stop_words]

# Classification of words as Positive, Negative, Neutral
for word in filtered_tokens:
    b2 = TextBlob(word)
    sentiment = b2.sentiment
    if sentiment.polarity > 0:
        print(f"{word} - Positive")
    elif sentiment.polarity < 0:
        print(f"{word} - Negative")
    else:
        print(f"{word} - Neutral")
# Code test to get the sentiment of a financial news article
# Using the GoogleNews API
# Work in progress

from GoogleNews import GoogleNews
from textblob import TextBlob

# Initialisation
googlenews = GoogleNews()
googlenews.set_lang('fr')
googlenews.set_period('14d')

# Keyword search
googlenews.search('finance')

# Récupération des résultats
# Getting the results
result = googlenews.result()
for news in result:
    print("-" * 50)
    print("Titre:", news['title'])
    print("Date:", news['date'])
    print("Description:", news['desc'])
    analysis = TextBlob(news['title'])
    sentiment = analysis.sentiment.polarity
    print("Sentiment:", sentiment)

analysis = TextBlob("SHIT YOU FUCKING ASSHOLE")
print("Sentiment:", analysis.sentiment.polarity)

googlenews.clear()
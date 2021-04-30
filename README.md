# Twitter Sentiment Analysis
<img src="figures/banner.png">



http://help.sentiment140.com/home

## Exploratory Data Analysis

Notebook Link: 

Key findings:



## Walk Through the Project



### Dataset

```
kaggle datasets download -d kazanova/sentiment140

unzip sentiment140.zip -d data/
```

Manually download data from: https://www.kaggle.com/kazanova/sentiment140



### Data Preprocessing

- Drop unnecessary columns, and only keep label and text
- Tokenize the text input with TweetTokenizer (from nltk package, specialized for tweets, such as preserving hashtags)
- Lemmatization that extracts the stem of words and parts of speech
- Drop stop words
- Normalize English slangs or abbreviations
- Remove "noises" (HTML tags, special characters, URLs, user mentions ie. "@someone")



## Modeling

Notebook Link: 



### Naive Bayesian Model



### RNN-LSTM












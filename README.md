# Twitter Sentiment Analysis
<img src="figures/banner.png">



In this project I want to compare two models **Naïve Bayes** and **RNN-LSTM** to conduct Sentiment Analysis . Sentiment analysis is a branch of Natural Language Process that analyses a given piece of text and predicts whether this piece of text expresses positive or negative sentiment.

The dataset is named sentiment140, created by researcher from Sandford University. The link for their dataset: http://help.sentiment140.com/home

The way they collected tweets is described in this paper: https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf, basically what they did, quoting their own words is:

> *Our approach was unique because our training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. We used the Twitter Search API to collect these tweets by using keyword search*



### The models results as follows:

**Naïve Bayes:** (clearly overfitting, bad performance)

```
Accuracy on train data: 0.9300671875
Accuracy on test data: 0.6442375

Confusion matrix:

Predicted	0	1	All
Actual			
0	173874	4536	178410
1	109308	32282	141590
All	283182	36818	320000
```



**RNN-LSTM** (more consistent results than Naïve Bayes)



No regularization (dropout), model overfits, validation accuracy around 75%

<img src="figures/lstm-no-dropout.png">

With regularization (dropout layers), improved slightly, 78% accuracy

<img src="figures/lstm-w-dropout.png">

Furthering cleaning noisy words, best model, 78.5% at 4th epoch

<img src="figures/lstm-clean-text.png">







## Walk Through the Project

### Dataset

```
kaggle datasets download -d kazanova/sentiment140

unzip sentiment140.zip -d data/
```

Manually download data from: https://www.kaggle.com/kazanova/sentiment140



### Exploratory Data Analysis

Notebook Link: https://georgehua.github.io/twitter-sentiment-analysis/EDA.html

Key findings:



### Data Preprocessing

- Drop unnecessary columns, and only keep label and text
- Tokenize the text input with TweetTokenizer (from nltk package, specialized for tweets, such as preserving hashtags)
- Lemmatization that extracts the stem of words and parts of speech
- Drop stop words
- Normalize English slangs or abbreviations
- Remove "noises" (HTML tags, special characters, URLs, user mentions ie. "@someone")



## Modeling

Notebook Link: https://georgehua.github.io/twitter-sentiment-analysis/Modeling.html



### Naive Bayesian Model



### RNN-LSTM





## Further Work

- Further data cleaning and re-labelling. As the data origin is from twitter, it is expected to contain a wide range of not "official" English words, so data cleaning is crucial in such a scenario. Furthermore, as the data labelling has been done automatically based on the reactions of the tweet, this labelling is by no means perfect and a human re-labelling of the whole data would certainly be beneficial.
- Introduce a neutral class, transforming the problem to a multi-class classification problem.
- Try out several other word embeddings or model architectures.
- Augment the data by diversifying it in order to make the model more robust, especially against sarcasm.



## Project Structure



    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Github Pages documents & figures
    │
    ├── figures             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebook          <- Jupyter notebooks for EDA and experiments
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`


------------










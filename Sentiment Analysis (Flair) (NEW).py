# %%
import numpy as np
import pandas as pd
import random, re, string
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from segtok.segmenter import split_single
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.graph_objects as go
import plotly.express as px

# %%
def Sentiment(df):
    if df['Score'] > 0.1:
        val = "POSITIVE"
    elif df['Score'] >= -0.1:
        val = "NEUTRAL"
    else:
        val = "NEGATIVE"
    return val

# %%
def remove_noise(comment, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(comment):
        token = re.sub(r'\W', ' ', token)
        token = re.sub(r'\s+[a-zA-Z]\s+', ' ', token)
        token = re.sub(r'\^[a-zA-Z]\s+', ' ', token)
        token = re.sub(r'\s+', ' ', token, flags=re.I)
        token = re.sub(r'^b\s+', '', token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# %%
def Clean(token):
        token = re.sub(r'\W', ' ', token)
        token = re.sub(r'\s+[a-zA-Z]\s+', ' ', token)
        token = re.sub(r'\^[a-zA-Z]\s+', ' ', token)
        token = re.sub(r'\s+', ' ', token, flags=re.I)
        token = re.sub(r'^b\s+', '', token)
        token = re.sub(' +', ' ', token)
        return token

# %%
"""
## Data Cleaning
"""

# %%
data_com = pd.read_excel(r"C:\Users\pvelazquez\OneDrive - ProKarma Softech Private Limited\Sentimental Analysis\PKeNPSGlobalResultsPivot.xlsx")
data = data_com[['ID', 'Survey', "Why would or wouldn't you recommend PK to a friend or colleague?"]]
data.columns = ['ID', 'Quarter', 'Comments']
data['ID'] = data['ID'].astype(int)
data['ID'] = data['ID'].astype(str)
data['Comments'] = data['Comments'].astype(str)


# %%
d_comments = pd.DataFrame(data["Quarter"])
d_comments["Comments"] = data["Comments"]
for i in range(len(d_comments['Comments'])):
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("n't", " not")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'s", " is")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'ll", " will")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'d", " would")


# %%
comments = {}
for i in range(len(data)):
    comments[str(i) + " - " + d_comments["Quarter"][i]] = split_single(d_comments["Comments"][i].lower())
sentences = [sentence for l in list(comments.values()) for sentence in l]
ds = pd.DataFrame(sentences, columns = ["Comments"])
quarter = []
for i in list(comments.items()):
    for j in range(len(i[1])):
        quarter.append(i[0].split(" "))
dq = pd.DataFrame(quarter, columns = ["ID", "-", "Quarter"])
ds = pd.concat([ds, dq], axis = 1)
ds = ds.drop(["-"], axis = 1)
comments = []
for comment in ds["Comments"]:
    comments.append(word_tokenize(str(comment).lower()))
clean_comments = []
for comment in comments:
    clean_comments.append(remove_noise(comment))

c = list()
for l in clean_comments:
    t = " ".join(l)
    t = re.sub(' +', ' ', t)
    c.append(t)


# %%
comments = {}
for i in range(len(data)):
    comments[i] = split_single(d_comments["Comments"][i].lower())
sentences = [sentence for l in list(comments.values()) for sentence in l]
id_c = []
for i in list(comments.items()):
    for j in range(len(i[1])):
        id_c.append(i[0])
ds = pd.DataFrame(id_c, columns = ["ID"])
ds["Comments"] = pd.Series(sentences)


# %%
"""
## Flair Model
"""

# %%
dc = pd.DataFrame(c, columns = ["Comments"])
score_list = list()
classifier = TextClassifier.load('en-sentiment')
for comment in dc["Comments"]:
    if comment != "nan" and comment != " " and comment != "":
        sentence = Sentence(comment)
        classifier.predict(sentence)
        score_list.append(str(sentence.labels).split(" "))
    else:
        score_list.append(["NEUTRAL", 0.0])
dc = pd.concat([dc, pd.DataFrame(score_list, columns = ["Sentiment", "Score"])], axis = 1)
for i in range(len(dc)):
    dc["Sentiment"][i] = Clean(dc["Sentiment"][i]).replace(" ", "")
    s = Clean(str(dc["Score"][i])).replace(" ", "")
    dc["Score"][i] = float(s[:1] + "." + s[1:])
    if dc["Sentiment"][i] == "NEGATIVE":
        dc["Score"][i] = -1*dc["Score"][i]
dc["Score"] = dc["Score"].astype(float)
dc["ID"] = ds["ID"].astype(int)
SC = dc[["ID", "Score"]].groupby(["ID"]).mean().reset_index()
dcfl = d_comments
dcfl["Score"] = SC["Score"]
dcfl['Sentiment'] = dcfl.apply(Sentiment, axis=1)


# %%
SCfl = dcfl[["Quarter", "Sentiment", "Score"]].groupby(["Sentiment", "Quarter"]).size().reset_index()
SCfl.columns = ['Sentiment','Quarter', 'Freq']


# %%
fig = px.bar(SCfl, x = "Quarter", y = "Freq", color = 'Sentiment', barmode = 'group', text = 'Freq')
fig.show()

# %%
"""
## Category
"""

# %%
cat = {"Benefits" : ["salary", "benefits", "compensation", "refunds"],
      "Environment" : ["culture", "life", "place", "co-workers", "organization", "team"],
      "Work": ["pressure", "load", "time"],
      "Leaders" : ["boss", "leader", "leadership", "desition"],
      "Growing" : ["grow", "skills", "carreer", "development", "opportunities"]}


# %%
a = []
for comment in dcfl["Comments"]:
    categories = list()
    for i in range(len(list(cat.items()))):
        for j in range(len(list(cat.items())[i][1])):
            if list(cat.items())[i][1][j] in comment.lower():
                categories.append(list(cat.items())[i][0])
                break
    a.append(categories)
dcfl["Categories"] = pd.Series(a)
dcfl = dcfl.reset_index()
dcfl['ID'] = dcfl['index'] + 1
dcfl = dcfl[['ID', 'Quarter', 'Comments', 'Score', 'Sentiment', 'Categories']]


# %%
dcfl.to_excel(r"C:\Users\pvelazquez\OneDrive - ProKarma Softech Private Limited\Sentimental Analysis\Flair_Results.xlsx")

# %%

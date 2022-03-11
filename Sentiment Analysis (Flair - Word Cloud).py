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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# %%
def Sentiment(df):
    if df['Score'] > 0:
        val = "POSITIVE"
    elif df['Score'] == 0:
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
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

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
data_com = pd.read_excel('C:/Users/ogarcia/OneDrive - ProKarma Softech Private Limited/DataScience/Data Sets/PK eNPS Global Results (2).xlsx')
data = data_com.drop(["Start time", "Completion time", "Email", "Name"], axis = 1)
data.columns = ["ID","Quarter", "How are you?", "Country", "Current Role/Focus", "Customer", "Vertical", "Service Line", 
               "Manage Employees", "Time in PK", "Recommend PK", "Comments", "Work Engaging", "Team Memebers", "Leadership",
              "Decisions", "Skills and Career", "Feedback", "Work Environment", "PK Family", "Core"]
data['Comments'] = data['Comments'].astype(str)
for i in range(len(data)):
    data["Quarter"][i] = data["Quarter"][i][7:]
    data["Quarter"][i] = data["Quarter"][i].replace(" ","-")
    if data["Country"][i] == "USA":
        data["Country"][i] = "USA/Canada"
data

# %%
d_comments = pd.DataFrame(data["Quarter"])
d_comments["Recommend PK"] = data["Recommend PK"]
d_comments["Comments"] = data["Comments"]
for i in range(len(d_comments['Comments'])):
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("n't", " not")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'s", " is")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'ll", " will")
    d_comments['Comments'][i] = d_comments['Comments'][i].replace("'d", " would")
d_comments

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
clean_comments
c = list()
for l in clean_comments:
    t = " ".join(l)
    t = re.sub(' +', ' ', t)
    c.append(t)
c

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
ds

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
dcfl

# %%
"""
## Word Cloud
"""

# %%
pos = dcfl[dcfl['Sentiment'] == 'POSITIVE']
pos

# %%
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
stop_words.update(["nan", " ", "pk", "would", "like", "not"])
comments = []
for comment in pos["Comments"]:
    comments.append(word_tokenize(str(comment).lower()))
clean_comments = []
for comment in comments:
    clean_comments.append(remove_noise(comment, stop_words))
clean_comments

# %%
len(clean_comments)

# %%
all_words = list(get_all_words(clean_comments))
all_words

# %%
nltk.FreqDist(set(all_words)).tabulate(20)

# %%
WF = nltk.FreqDist(all_words)
WF.tabulate(10)

# %%
WFdf = pd.DataFrame(WF.most_common(20))
WFdf.columns = ['Word', "Frequency"]
fig = px.bar(WFdf, x = "Frequency", y = "Word", orientation='h', text = "Frequency")
fig.show()

# %%
text = nltk.Text(all_words)
finder = nltk.collocations.BigramCollocationFinder.from_words(text)
df_pos = pd.DataFrame.from_dict(finder.ngram_fd.most_common(25))
df_pos.columns = ["Words", "Frequency"]
comments = []
for l in df_pos["Words"]:
    comments.append(" ".join(l))
df_pos["Words"] = pd.Series(comments)
fig = px.bar(df_pos, x = "Frequency", y = "Words", orientation='h', text = "Frequency")
fig.show()

# %%
df_pos['Sentiment'] = 'Positive'
df_pos

# %%
neg = dcfl[dcfl['Sentiment'] == 'NEGATIVE']
neg

# %%
stop_words = set(stopwords.words('english'))
stop_words.update(["nan", " ", "pk", "would", "like"])
comments = []
for comment in neg["Comments"]:
    comments.append(word_tokenize(str(comment).lower()))
clean_comments = []
for comment in comments:
    clean_comments.append(remove_noise(comment, stop_words))
clean_comments

# %%
all_words = list(get_all_words(clean_comments))
all_words

# %%
WF = nltk.FreqDist(all_words)
WF.tabulate(10)

# %%
WFdf = pd.DataFrame(WF.most_common(20))
WFdf.columns = ['Word', "Frequency"]
fig = px.bar(WFdf, x = "Frequency", y = "Word", orientation='h', text = "Frequency")
fig.show()

# %%
text = nltk.Text(all_words)
finder = nltk.collocations.BigramCollocationFinder.from_words(text)
df_neg = pd.DataFrame.from_dict(finder.ngram_fd.most_common(25))
df_neg.columns = ["Words", "Frequency"]
comments = []
for l in df_neg["Words"]:
    comments.append(" ".join(l))
df_neg["Words"] = pd.Series(comments)
fig = px.bar(df_neg, x = "Frequency", y = "Words", orientation='h', text = "Frequency")
fig.show()

# %%
df_neg['Sentiment'] = 'Negative'
df_neg

# %%
df_sen = pd.concat([df_pos, df_neg]).reset_index(drop = True)
df_sen

# %%
df_sen.to_excel('C:/Users/ogarcia/OneDrive - ProKarma Softech Private Limited/DataScience/Data Sets/Word_Cloud.xlsx')
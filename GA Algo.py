from google.colab import drive
drive.mount('/content/drive')

!ip install fitz
!pip install pymupdf
!pip install path

!pip install gaft

import pandas as pd
import numpy as np
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

path="/content/eval.xlsx"
df=pd.read_excel(path)

print(df.head())

import nltk
from nltk.corpus import stopwords
nltk.download('words')
nltk.download("popular")
nltk.download('stopwords')

import fitz
import pandas as pd
from path import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import OrderedDict

def clean (text):

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+\www\.\S+', '', text)

    text = re.sub('<.*?>+','', text)

    text = re.sub('[%s]' % re.escape(string.punctuation),'' , text)
  
    text = re.sub('\n', '', text)

    text = re.sub('\w\d\w*', '', text)

    text=[word for word in text.split(' ') if word not in stopword]

    text=" ".join(text)

    text = [stemmer.stem (word) for word in text.split(' ')]

    text=" ".join(text)

    return text

df["text"] = df["text"].apply(clean) 
print(df.head())

X = df['text']
y = df['HS']
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

train_size = int(0.7 * X_vec.shape[0])
X_train = X_vec[:train_size]
y_train = y[:train_size]
X_test = X_vec[train_size:]
y_test = y[train_size:]

indv_template = BinaryIndividual(ranges=[(0, 1)] * X_train.shape[1])
population = Population(indv_template=indv_template, size=25)
population.init()

selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population,selection = RouletteWheelSelection(),crossover =UniformCrossover(pc=0.8, pe=0.5),mutation =FlipBitMutation(pm=0.1))


# Define the fitness function
@engine.fitness_register
def fitness_function(indv):
    weights = np.array(indv.solution)
    clf = DecisionTreeClassifier(random_state=0, max_depth=10, max_features=None)
    clf.fit(X_train.multiply(weights), y_train)
    y_pred = clf.predict(X_train.multiply(weights))
    a= -accuracy_score(y_train, y_pred).tolist()
    return a
  
  # Run the GA engine
engine.run(ng=10)

# Get the best individual
best_indv = engine.population.best_indv(engine.fitness)


# Evaluate the model
weights = np.array(best_indv.solution)
clf = DecisionTreeClassifier(random_state=0, max_depth=10, max_features=None)
clf.fit(X_train.multiply(weights), y_train)
y_pred = clf.predict(X_test.multiply(weights))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

def plotgraph(y_test, y_pred):
  plt.ylabel(("Predicted"))
  plt.xlabel("Actual")
  plt.scatter(y_test, y_pred, color = 'green', label = 'predicted values', s = 10)
  plt.axline([-1000,-1000], [1000,1000], label = 'actual values')
  plt.legend(loc = "upper left")
  plt.show()
  
  y_test
  
  ! pip install scikit-plot==0.3.7
  
  import scikitplot as skplt
  
  skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, title = 'Confusion Matrix for GA')
  
  skplt.estimators.plot_learning_curve(DecisionTreeClassifier(), X_test, y_test,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Hate Speech Classification Curve")
  
  fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix",
                                    cmap="Oranges",
                                    ax=ax1)

ax2 = fig.add_subplot(122)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Purples",
                                    ax=ax2)

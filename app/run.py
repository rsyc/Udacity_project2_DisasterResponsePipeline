import re
import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin



app = Flask(__name__)

def tokenize(text):
    '''
    This function is for tokenizing a given text 
    (eg. each message from X in the model). Steps include
    tokenizing sentences into words, lemmatizing (words root), 
    normalizing (all to lower case), removing stop words and empty spaces
    
    Input:
        text: a given text (a message at a time)
    output:
        clean_tokens: cleaned tokens to be used in the model        
    '''
    # tokenize text
    words = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in words:
        # lemmatize, normalize case, and remove leading/trailing white space
        tok = tok.lower()
        clean_tok = re.sub(r'[^a-zA-Z0-9]', ' ', lemmatizer.lemmatize(tok, pos='v')).strip()
        # remove stop words and empty strings
        if clean_tok != '' and clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)
     
    return clean_tokens

    
    #tokens = word_tokenize(text)
    #lemmatizer = WordNetLemmatizer()

    #clean_tokens = []
    #for tok in tokens:
    #    clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    #    clean_tokens.append(clean_tok)

    #return clean_tokens

class GenreExtractor(BaseEstimator, TransformerMixin):
    '''
    This is a custom Transfer function to separate genre
    data from the text data
    '''
    def get_genre_data(self, x):
        '''
        This function separates genre data and turn the one column
        of genre (in text) to columns of dummies each representing
        one gerne. These columns then will be used in the model for
        training and test.
        '''
        genres = [record[1] for record in x]
        genre_dummies = pd.get_dummies(genres)
        return genre_dummies
   

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply function to all values in X
        X_tagged = self.get_genre_data(X) 

        return pd.DataFrame(X_tagged)

class TextExtractor(BaseEstimator, TransformerMixin):
    '''
    This is a custom Transfer function to separate text
    data from the genre data
    '''
    def get_text_data(self, x):
        '''
        this function separates the text data and returns
        it to be used in the model for further analysis.
        '''
        return [record[0] for record in x]
      

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply function to all values in X
        X_tagged = self.get_text_data(X) 

        return X_tagged


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('processedData', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
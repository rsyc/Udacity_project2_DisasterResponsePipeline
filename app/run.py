import re
import json
import plotly
import pandas as pd
import numpy as np
import seaborn as sns

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

from app.WordCloud_process import wordcloud_generator
import plotly.express as px

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
        This function separates genre data and turns this categorical 
        column of genre (in text) to a numerical column: each number 
        representing one gerne. This column then will be used in the model 
        for training and test. 
        '''
        genres = [record[1] for record in x]
        replaced_bynum = [1 if x=='direct' else 2 if x=='news' else 3 for x in genres]
        return replaced_bynum 
   

    def fit(self, x, y=None):
        '''
        The fit function of the transformer.
        The function does the fitting. For our class this function does not
        do anything speciall. Most of work is done in the transformation. 
        input: x variables
        output: self
        '''
        return self

    def transform(self, X):
        '''
        Transformation function
        Input: X variable
        output: dataframe
        '''
        # apply function to all values in X
        X_tagged = self.get_genre_data(X) 

        return pd.DataFrame(X_tagged) # np.array(X_tagged).reshape(-1, 1)

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
        '''
        The fit function of the transformer.
        The function does the fitting. For our class this function does not
        do anything speciall. Most of work is done in the transformation. 
        input: x variables
        output: self
        '''
        return self

    def transform(self, X):
        '''
        Transformation function
        Input: X variable
        output: dataframe
        '''
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
    
    # Count number of tagged cases in each category and show them in a bar chart after 
    # grouping by their genre
    plot_data = df.drop(['id', 'message', 'original'], axis=1).groupby(by=['genre']).mean() 
    # second bar graph
    fig2 = px.bar(plot_data, title="percentage of values for each category")
        
    # separating dataframe based on different genre 
    Direct_group = df[df['genre']=='direct']
    News_group = df[df['genre']=='news']
    Social_group = df[df['genre']=='social']
    
    # calling wordcloud_generator function for each genre-separated dataframe
    # to run steps on the messages in each group and generate a word cloud for each.
    Direct_wordcloud = wordcloud_generator(Direct_group, 'message')
    News_wordcloud = wordcloud_generator(News_group, 'message')
    Social_wordcloud = wordcloud_generator(Social_group, 'message')
    
    # Visualize the word cloud for each dataframe
    Direct_wordcloud_fig = px.imshow(Direct_wordcloud.to_image())
    News_wordcloud_fig = px.imshow(News_wordcloud.to_image())
    Social_wordcloud_fig = px.imshow(Social_wordcloud.to_image())
    
    # defining the layout for each word cloud
    Direct_wordcloud_fig.update_layout(
        title=dict(text='150 most common words in Disater Scenario with genre=Direct', x=0.5),
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        hovermode=False
    )
    News_wordcloud_fig.update_layout(
        title=dict(text='150 most common words in Disater Scenario with genre=News', x=0.5),
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        hovermode=False
    )
    Social_wordcloud_fig.update_layout(
        title=dict(text='150 most common words in Disater Scenario with genre=Social', x=0.5),
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        hovermode=False
    )
    
    
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
    
    # Appending figures that are generated above to the graphs
    graphs.append(fig2)
    graphs.append(Direct_wordcloud_fig) 
    graphs.append(News_wordcloud_fig)
    graphs.append(Social_wordcloud_fig)
    
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
    query = query.split("| ")
    print(query)

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
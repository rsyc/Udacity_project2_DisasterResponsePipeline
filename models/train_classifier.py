import sys
import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy.core.multiarray
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OrdinalEncoder

def load_data(database_filepath):
    '''
    This function is to import data from .db files.
    inout:
        database_filepath: path to the .db file
    Output:
        X: Extracted varibale columns (messages, genre) 
        Y: Extracted category columns
        category_names: column name of the categories
    '''
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql_table('processedData', con=engine)
    X = df[['message','genre']].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns 
    #en = OrdinalEncoder()
    #print(np.unique(en.fit_transform(np.array(df['genre']).reshape(-1, 1))))
    
    return X, Y, category_names
    
    
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

# Create Function Transformer to use Feature Union
class GenreExtractor(BaseEstimator, TransformerMixin):
    '''
    This is a custom Transfer function to separate genre
    data from the text data
    '''
    def get_genre_data(self, x):
        '''
        This function separates genre data and turn this categorical 
        column of genre (in text) to a numerical column: each number 
        representing one gerne. This column then will be used in the model 
        for training and test.
        '''
        genres = [record[1] for record in x]
        #genre_dummies = pd.get_dummies(genres)
        replaced_bynum = [1 if x=='direct' else 2 if x=='news' else 3 for x in genres]
        return replaced_bynum #return genres
   

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply function to all values in X
        X_tagged = self.get_genre_data(X) 
        return pd.DataFrame(X_tagged) #np.array(X_tagged).reshape(-1, 1) #pd.DataFrame(X_tagged)

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



def build_model():
    '''
    This function is to make a model. We make the model
    first by defining a pipeline that encloses all the
    transforming steps to make the features ready for
    the estimator/predictor. 
    
    The transformation is done in 2 parallel steps:
    1) to the text column (messages) to turn them into 
    vectors for the model to use.
    2) to the genre column to turn it into dummy columns
    
    The result of both steps are then concatenated together 
    and fed to the model. As the model should do a 
    multi-classification based on the input features, 
    a RandomForestClassifier is used inside a MultiOutputClassifier
    to fit the classification per target.
    
    The pipeline and the parameters that we need to tune in the model
    are then given to a GridSearchCV to do the cross validation.
    
    Input: Non
    Output: a tuned and optimized classification model (cv)
    '''
    pipeline3 = Pipeline([
        ('features', FeatureUnion([
                 ('text_features', Pipeline([
                    ('txt_ext', TextExtractor()),
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('genre_features', Pipeline([
                    ('gen_ext', GenreExtractor()),
                #    ('le' , OrdinalEncoder())
                ]))            
             ])),
        #('emblance', Multilass_Balencer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # BalancedRandomForestClassifier(replacement=True)
    ])
    
    parameters = {
        'clf__estimator__max_features': ['log2', 'sqrt'],
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_leaf': [5, 20]
    }

    cv = GridSearchCV(pipeline3, param_grid = parameters, cv=2)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function is to evaluate our model.
    This is done by showing the accuracy, precision, and recall of the tuned model.
    Input: 
        model: the tuned model
        X_test: variables for teting the model
        Y_test: known values of classes for the X_test
        category_names: names of the calsses/categoreis
    Output: Non, it print out the F1-score, precision, and recall
    '''
    y_pred_cv = model.predict(X_test)
    # iterating through the columns and report the f1 score, 
    # precision and recall for each output category of the dataset.
    for j in range(len(category_names)):
        print(classification_report(Y_test.transpose()[j], y_pred_cv.transpose()[j]))


def save_model(model, model_filepath):
    # Export your model as a pickle file and save it to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_params_)

        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
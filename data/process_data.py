import sys
import pandas as pd
import time
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads data from csv files.
    Input: 
        messages_filepath: the path to the csv 
        file where message data is saved. 
        categories_filepath: the path to the csv 
        file where category data is saved.
    Output: gives a merged data set from messages 
        and categories datasets.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df   


def clean_data(df):
    '''
    This function cleans the input dataframe. 
    The cleaning includes, splitting the text data
    in the "categories" column into separate column 
    (36 columns), converting category values to binary 
    values, and removing duplicates.    
    Input: 
        df: input dataframe which is obtained from 
        merging "messages" and categories" datasets
        
    Output: 
        df: cleaned dataframe    
    '''
    # create a dataframe using the data in the "categories" column
    # separated by ; and exapnd them into 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    sl = slice(-2)
    category_colnames = list(map(lambda x: x[sl] , row ))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str.split("-").str.get(1) #.str.extract('(\d+)') # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column]) # convert column from string to numeric
        
        
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1) # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe
    
    
    # Remove duplicates
    #df.duplicated().value_counts() # check number of duplicates
    #df[df.duplicated()]
    df = df.drop_duplicates() # drop duplicates
    #df.duplicated(subset=['message']).value_counts() # check duplicates in "message" column
    
    # More investigations on the "messages" duplicates showed that
    # the only differences are usually in the categories, for example
    # "medical_help" is 1 in one case and 0 in the other case. In this 
    # particular example, it seems that it should not be categorized as
    # medical_help as the message shows, so there might have been a problem,
    # possibly user mistake! so we removed such occurances:
    df = df.drop_duplicates(subset=['message','id'])
    
    # here we see that the 3 cases with different id but same "message"
    # have in fact no message: "#NAME?   NaN". These rows would not be helpful 
    # for our model as there is no message to be used for training/testing
    # as there is only 3 cases like this and dropping them wont harm our 
    # model, I am going to drop them
    #df[df.duplicated(subset=['message'])]
    df = df.drop_duplicates(subset=['message'])
    
    
   # print('total size of unique rows in df \n', df.nunique()) # check number of duplicates
    # it seems that we still have 3 different values for "rlated" category
    # The value "2" does not make sense. We only have to have 
    # 0=not-related and 1=rela. So we remove rows with the value of related=2
    df = df.drop(df[df['related']==2].index)   

    return df

def save_data(df, database_filename):
    '''
    This function saves and exports the input dataframe into a sqlite database.
    Input:
        df: input dataframe
        database_filename: file path and name for the sql .db file
    Output: Non
    '''
    engine = create_engine('sqlite:///'+str(database_filename))
    df.to_sql('processedData', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
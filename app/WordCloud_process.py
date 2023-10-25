from wordcloud import WordCloud
from nltk.corpus import stopwords 

def wordcloud_generator(df, column_name):
    '''
    This function is to do generate word cloud on a given column
    of an input dataframe.
    input: 
        df: input dataframe
        column_name: the name of the column that we will extract 
            text to feed to our word cloud generator
    output:
        wordcloud: a word cloud generated from all the text in our
            dataframe column
    '''
    # Join the different text/message from each row together separated by ",".
    long_string = ','.join(list(df[column_name].values)) # a long string including all the messages
    # define stop wrods that we want to remove from our text to make it clean
    stop_words = list(WordCloud().stopwords)
    stop_words = stop_words + stopwords.words("english")
    # Create a WordCloud object
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud = wordcloud.generate(long_string)
    
    return wordcloud
from wordcloud import WordCloud
from nltk.corpus import stopwords 

def wordcloud_generator(df, column_name):
    # Join the different processed titles together.
    long_string = ','.join(list(df[column_name].values))
    # Create a WordCloud object
    stop_words = list(WordCloud().stopwords)
    stop_words = stop_words + stopwords.words("english")
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud = wordcloud.generate(long_string)
    
    return wordcloud
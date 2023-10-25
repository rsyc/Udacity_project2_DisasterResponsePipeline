# Disaster Response Pipeline Project

## Introduction:
Here I have analyzed disaster data from [Appen](https://appen.com/) (formally Figure 8) to build a model for an Application Programming Interface (API) that classifies disaster messages. The dataset used for building this model, training and testing it includes real messages that were sent during disaster events.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Before running the app make sure you have installed all the needed packaged and/have the correct versions (check the "Needed libraries" section)

3. While you are in the project directory run your web app: `python app/run.py`

## File/Folder structure:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
|- WordCloud_process.py # python code to make word cloud of a dataframe

- data
|- disaster_categories.csv  # data to process (categories information)
|- disaster_messages.csv  # data to process (messages and their genre)
|- process_data.py
|- DisasterResponse.db   # database with saved clean data

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

## Needed libraries:
The code needs the libraries as listed above.

For making the plots in the flask using plotly we need to install this version of plotly

- ` pip install plotly==5.11.0 `

For the wordcloud to work you need to install pillow and upgrade wordcloud, pip and pillow:

- ` pip install --upgrade wordcloud==1.8.0 `
- ` pip install Pillow`
- ` pip install --upgrade Pillow `
- ` pip install upgrade pip`

## App description:

1) The first graph is a visualization of the Distribution of Message Genres. It shows the number of counts of each genre. 

2) The second graph is the representation of the percentage of True values (==1) for each category within each genre. As you may see from the plot (for our dataset) the percentage of category named "related" is the highest (if you hover over each colored section you can see the information regarding variable name, genre, and the normalized count/percentage value). 
	- For our specific data the percentage of messages categorized as "related" are 67% in "direct" genre, 82% in "news", and 87% in "social" genre. The percentage of other categories are considerably lower. 
	- This plot shows that the number of flags across all categories are imbalanced. Fixing that is a challenging task cause this is a multiclass classification problem, meaning that each row in the data can contain flags from multiple categories. Available methods to deal imbalanced data are only available for binary classification, not for multi-class classification. Also writing a custom function that deals with it is challenging as we need to feed both variables and targets to it while this is not possible in pipeline (x and y values are only given to the last step where the prediction/estimation of the model happend). 
    - This imbalanceness causes overfitting problem, which is why you may see a great tendency for the model to classify you messages as only "related". 
    
3) The last three graphs show the word cloud representation from the messages separated by their genre. 
	- The word cloud for the genre=direct shows that the most frequent words used in this genre include "Please", "help", "need", "Thank", "people", "want", "know", and "information". This shows that in these "direct" messages people are "directly" asking for help and/or some information.
    - The word cloud for the genre=news shows that the most frequent words used in this genre include "people", "area", "water", "said", "country", "region", "government", "food", and "flood", Which reflects news of some problem/concerns at regions and/or countries regarding water, food, flood etc.
    - The word cloud for the genre=seocial shows that the most frequent words used in this genre include "sandy", "Co (possibly company)", "earthquake", "Haiti", "Santiago", and "storm". This word cloud, as oppose to the other two word clouds, does not highlight a specific topic. Wheras it seems that this word cloud represent genral messages that people have given.
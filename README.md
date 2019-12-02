# Disaster Response Pipeline Project

# Installations:
To Run the project, you need to import some libraraies to help you in doing some tasks: 
1. Pandas   https://pandas.pydata.org/pandas-docs/stable/
2. numpy    https://numpy.org/
3. sqlalchemy   https://docs.sqlalchemy.org/en/13/
4. Pipeline
5. WordNetLemmatizer
6. word_tokenize
7. MultiOutputClassifier
8. GridSearchCV

# Project Modules: 
## Extract, Transform and Load Module.
1. Loading the datasets from messages and categories. 
2. Cleaning the data by merging, remove duplicates, and converting to integers. 
3. Loading into the database. 

## Machine Learning Pipeline.
1. Read Data from Database
2. Tokenization function to process your text data
3. Build a Machine Learning Pipeline
4. Train the pipeline
5. Testing the Model
6. Improve the model
7. Retesting the system to improve the results. 

## Web Application Interface. 
1. Categorize the content of the database which came from people during crisis.
2. Visualize the result for better view for the end user.

## Backgroun
This application built as requirments for the Udacity Nano Degree which testing what we learnt in the Software engineering and Data Enginering concepts. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

import sys
import pandas as pd
import numpy as np
import sqlalchemy as db

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    db_string = 'sqlite:///{}'.format(database_filepath)
    engine = db.create_engine(db_string)
    connection = engine.connect()
    metadata = db.MetaData()
    message_mapping = db.Table('messages_cats', metadata, autoload=True, autoload_with=engine)
    query = db.select([message_mapping])
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    df = pd.DataFrame(ResultSet)
    df.columns = ResultSet[0].keys()
#     df
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(Y_train):
    corr = Y_train.corr()
    corr_sorted = []
    for col in corr.loc['related'].sort_values(ascending=False).index:
        corr_sorted.append(Y_train.columns.get_loc(col))
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('model', ClassifierChain(LogisticRegression(max_iter=300), order=corr_sorted))
                    ])
    
    parameters = {
        'model__base_estimator__class_weight': (None,'balanced'),
        'model__base_estimator__solver': ('lbfgs','liblinear')
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print("{}\n{}".format(col, classification_report(Y_test[col], Y_pred[:,i])))
        
    pr_score = precision_score(Y_test, Y_pred, average='micro')
    re_score = recall_score(Y_test, Y_pred, average='micro')
    print("Overall Precision Score: {}\n Overall Recall Score: {}". format(pr_score, re_score))

def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
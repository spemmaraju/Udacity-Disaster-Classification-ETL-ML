import sys
import pandas as pd
import numpy as np
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on = 'id')
    # create a dataframe of the 36 individual category columns
    cat_list = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row_first = cat_list.iloc[0]
    # use this row to extract a list of new column names for categories.
    cat_colnames = row_first.str.split('-', expand=True)[0]
    # rename the columns of `categories`
    cat_list.columns = cat_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in cat_list:
        cat_list[column] = cat_list[column].str.split('-', expand=True)[1]
    # set each value to be the last character of the string
        
    # convert column from string to numeric
        cat_list[column] = pd.to_numeric(cat_list[column])
    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, cat_list], axis = 1)
    
    return df


def clean_data(df):
    # drop duplicates across all rows
    df.drop_duplicates(inplace=True)
    # drop all rows with field "related" = 2 because they do not have any of the other columns populated and are 
    # probably messages that could not be classified
    df.drop(df[df['related']==2].index, inplace=True)
    # drop the column "child_alone" because it takes only one value (0) and thus has no predictive power
    df.drop(columns='child_alone', inplace=True)
    # Identify messages that are duplicates of each other
    unique_messages, count = np.unique(df['message'], return_counts=True)
    duplicate_messages = []
    for i in range(len(unique_messages)):
        if count[i] > 1:
            duplicate_messages.append(unique_messages[i])
    # We see from the data that there are 4 rows with a meaningless "#NAME?" message. We drop these 4 rows
    df.drop(df[df['message']=="#NAME?"].index, inplace=True)
    # For the rest of the duplicate messages, we see that even with same "ID", the number of disaster type categories are different
    # We will retain the row which has been classified into more number of categories and drop the other
    # Retain the row which has the highest number of categories for each duplicate message
    df = df.groupby('id').max().reset_index()
    return df

def save_data(df, database_filename):
    # Enter the name of the database
    db_string = 'sqlite:///{}'.format(database_filename)
    engine = sqlalchemy.create_engine(db_string)
    # Enter the name of the SQL table to be stored in the database
    df.to_sql('messages_cats', engine, index=False, if_exists = 'replace')  


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
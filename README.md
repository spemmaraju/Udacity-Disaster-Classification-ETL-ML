# Udacity-Disaster-Classification-ETL-ML

## Purpose of the project
The purpose of this project is to generate a web app that can classify disaster messages. Using the app an individual can enter the message in the app. It will automatically classify the message into different categories, which can then be used to escalate it to relevant authorities or take appropriate action. 

## Motivation
In the middle of a natural disaster, several thousand messages are generated from various sources including social media. In order to be more responsive and quickly identify crucial needs, we must be able to rapidly classify these messages into different categories and assign responsibilities to each category. This can save a lot of time and confusion. 

## Folder Structure of the repository

├── .ipynb_checkpoints                   # Checkpoints for the jupyter notebook files
├── app                    		 # Contains all the relevant files to generate a web app
├── data                     		 # Contains the source dataset, SQL database and data cleaning code 
├── models                    		 # Machine learning model code and saved final model file
├── .gitattributes.txt                   # A file to set attributes
├── DisasterDB.db			 # SQL Database generated by the Jupyter Notebook
├── ETL Pipeline Preparation.ipynb       # Jupyter Notebook for ETL pipeline preparation
├── finalized_model.sav                  # Final model saved by Jupyter notebook
├── ML Pipeline Preparation.ipynb        # Jupyter Notebook for ML pipeline preparation
└── README.md                            # File to store details about the project

## Libraries used:
1. NumPy, Pandas - For creating and manipulating dataframes and using in-built functions and features for data manipulation
2. Matplotlib - For generating plots and various data visualizations
3. sqlalchemy - To operate SQL commands in Python
4. NLTK - To perform NLP operations like tokenization, lemmatization etc.
5. Plotly - A library that helps generate interactive plots and visualizations, here used to generate choropleths
6. sklearn - To develop a machine learning model for the message categorization and perform appropriate parameter optimizationa and calculate appropriate scores
7. Pickle - To save the final model generated by the ML model
8. Flask - To build web applications in Python


## Instructions:
1. Navigate to the folder which has this readme file, start a new terminal and follow the instructions below.
	a) Run the ETL pipeline by entering - "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
	b) Run the ML pipeline by entering - "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"
2. Navigate to the "app" folder on the terminal and enter the following command - "python run.py"
3. Visit the link http://0.0.0.0:3001/ to see the web app. If on Windows on a local machine, try http://localhost:3001/

## Screenshots of the Web App:


## Key Observations on the model development process:
1. The dataset is imbalanced with some of the categories like aid_centers and fire having very few positives. Therefore, in the model development process, parameter tuning was performed by emphasizing on recall instead of precision. This may result in more false positives, but given the context, it is better to flag more categories for a disaster related message than to miss any important category due to poor recall. 
2. There is inherent hierarchy and correlation in the categories that are predicted. For example, related = 0 does not flag any other category while related = 1 flags several categories. Similarly, some of the other categories are strongly correlated. Exploiting this correlation can improve Recall. Hence I developed a ClassifierChain model instead of a MultiOutputClassifier
3. Accuracy of the model is very low. However, it is important to note that Accuracy doesn't make sense in a multilabel model because even if the model gets 9 out of 10 labels correctly and only misses one, in practice that's an excellent outcome. But the accuracy metric penalizes 9/10 and 0/10 the same.

## License
This app was created as a part of the Udacity Data Scientist Nanodegree

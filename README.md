# CONSUMER-COMPLAINTS
This repository contains a machine learning pipeline for classifying consumer complaint narratives into financial product categories using natural language processing (NLP) and supervised learning models. The goal is to automate the categorization of complaints for improved analysis and response.

Dataset

  Source: complaints.csv (mounted from Google Drive)
  
  Size: ~9.4 million entries, multiple columns
  
  Used columns: Product, Consumer complaint narrative
  
Categories mapped:

    0: Credit reporting, credit repair services, or other personal consumer reports
    
    1: Debt collection
    
    2: Consumer/Student/Payday/Vehicle loan
    
    3: Mortgage

Data Preprocessing

      Missing values removed for relevant columns
      
      Product categories mapped to integer labels
      
    Text cleaning:
    
      Lowercasing, punctuation and number removal
      
      Stopword removal and lemmatization (NLTK)
    
    Text length and class distribution visualized

Feature Engineering

    Text vectorized using TF-IDF (max 5000 features, unigrams and bigrams)

Models Implemented

  The notebook implements and evaluates the following machine learning models:
  
    1.Multinomial Naive Bayes
    
    2.Logistic Regression
    
    3.Linear Support Vector Classifier (LinearSVC)

Evaluation Metrics
  The models are evaluated using:

    Classification report
    
    Confusion matrix
    
    Accuracy score
    
    Cross-validation scores
Requirements
  To run this notebook, you'll need:
  
    Python 3.x
    
    Pandas
    
    NumPy
    
    Matplotlib
    
    Seaborn
    
    Scikit-learn
    
    NLTK

Note
  
    The Random Forest classifier was not implemented due to system constraints. Future work could include implementing this model for comparison.

    All NLP preprocessing steps are handled with NLTK.

Results

    Best performance: LinearSVC and Logistic Regression (accuracy ~0.90)
    
    Naive Bayes performed moderately well (accuracy ~0.85)
    
    Random Forest code present, but model results not completed or reported

Cross-Validation

    5-fold cross-validation performed on the best model (LinearSVC)
    
    Mean CV accuracy reported

Prediction Functionality

    Utility function provided to predict the category of new complaint texts using the trained model and preprocessing pipeline

Future Improvements

    Implement Random Forest classifier
    
    Experiment with deep learning models
    
    Perform hyperparameter tuning
    
    Address class imbalance if present
    
    Deploy as a web application
  


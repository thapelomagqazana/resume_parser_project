# import pandas as pd

# # Load the dataset
# dataset_path = 'data/UpdatedResumeDataSet.csv'
# df = pd.read_csv(dataset_path)

# # Display basic information about the dataset
# print("Dataset Information:")
# print(df.info())

# # Display the first few rows of the dataset
# print("\nSample Data:")
# print(df.head())

# # Display column names
# print("\nColumns:")
# print(df.columns)

# # Display data types
# print("\nData Types:")
# print(df.dtypes)

# # Check for missing values
# print("\nMissing Values:")
# print(df.isnull().sum())

# # Examine unique values in a specific column
# print("\nUnique Values in 'Category' column:")
# print(df['Category'].unique())

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample resume text
sample_resume = "This is a sample resume text with punctuation, stop words, and different tenses. Coding skills are important for the job."

# Tokenize the text
tokens = word_tokenize(sample_resume)

# Remove stop words and punctuation
stop_words = set(stopwords.words('english'))
filtered_tokens = [token.lower() for token in tokens if (token.isalpha() and token.lower() not in stop_words)]

# Perform stemming
porter_stemmer = PorterStemmer()
stemmed_tokens = [porter_stemmer.stem(token) for token in filtered_tokens]

# Perform lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in filtered_tokens]

# Display the results
print("Original Text:")
print(sample_resume)

print("\nTokenized Text:")
print(tokens)

print("\nFiltered Tokens (without stop words and punctuation):")
print(filtered_tokens)

print("\nStemmed Tokens:")
print(stemmed_tokens)

print("\nLemmatized Tokens:")
print(lemmatized_tokens)


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(job_resumes):

    processed_texts = []
    for resume in job_resumes:
        # Tokenize the text
        tokens = word_tokenize(resume)

        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token.lower() for token in tokens if (token.isalpha() and token.lower() not in stop_words)]
        
        # Peform lemmatization
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Joining tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        processed_texts.append(processed_text)

    return processed_texts

# Load the dataset
def load_dataset(dataset_path: str):
    return pd.read_csv(dataset_path)

def extract_keywords(dataFrames):

    # DataFrame contains the resumes and job titles. Examine unique values in a specific column
    unique_job_titles = dataFrames["Category"].unique()
    # print("\nUnique Values in 'Category' column:")
    # print(unique_job_titles)
    
    #Create a dictionary to store job-title-specific keywords
    job_title_keywords = {}

    for job_title in unique_job_titles:
        # Extract resumes for the current job title
        job_resumes = dataFrames[dataFrames["Category"] == job_title]["Resume"].tolist()
        # print(job_resumes)

        # Perform text processing (tokenization, stop word removal, etc.)
        processed_resumes = preprocess_text(job_resumes)
        # print(processed_resumes)

        # Use CountVectorizer to identify keywords
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(processed_resumes)

        # Get feature names (words) and their frequencies
        feature_names = vectorizer.get_feature_names_out()
        word_frequencies = X.sum(axis=0).A1

        # Identify top keywords based on frequencies
        top_keywords = [feature_names[idx] for idx in word_frequencies.argsort()[::-1][:10]]

        # Store the top keywords in the dictionary
        job_title_keywords[job_title] = top_keywords
    
    return job_title_keywords

def frequency_of_keyword(df):

    #  Combine all resumes into a single text for analysis
    all_resumes_text = ' '.join(df['Resume'])

    # Perform text processing (tokenization, stop word removal, etc.)
    processed_all_resumes = preprocess_text([all_resumes_text])

    # Use CountVectorizer to identify keywords and their frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_all_resumes)

    # Get feature names (words) and their frequencies
    feature_names = vectorizer.get_feature_names_out()
    word_frequencies = X.sum(axis=0).A1

    # Create a DataFrame to store keyword frequencies
    keywords_df = pd.DataFrame({"Keyword": feature_names, "Frequency": word_frequencies})

    # Sort the DataFrame by frequency in descending order
    keywords_df = keywords_df.sort_values(by="Frequency", ascending=False)

    # Display the top keywords and their frequencies
    print("Top Keywords and Frequencies:")
    print(keywords_df.head(10))

    generate_statistics(keywords_df)

def generate_statistics(keywords_df):
    # Calculate basic statistics
    total_keywords = len(keywords_df)
    unique_keywords = len(keywords_df["Keyword"].unique())
    average_frequency = keywords_df["Frequency"].mean()
    max_frequency = keywords_df["Frequency"].max()

    # Display basic statistics
    print("\nBasic Statistics:")
    print(f"Total Keywords: {total_keywords}")
    print(f"Unique Keywords: {unique_keywords}")
    print(f"Average Frequency: {average_frequency:.2f}")
    print(f"Maximum Frequency: {max_frequency}")

def basic_keyword_matching(resume_text, job_title_keywords, job_title):
    """
    Basic keyword matching to shortlist candidates.
    """

    # Get the list of job-title-specific keywords
    keywords = job_title_keywords.get(job_title, [])

    # Check if any of the keywords is present in the resume text
    matching_keywords = [keyword for keyword in keywords if keyword.lower() in resume_text.lower()]

    return len(matching_keywords) > 0

if __name__ == "__main__":
    
    dataset_path = 'data/UpdatedResumeDataSet.csv'
    df = load_dataset(dataset_path)
    job_title_keywords = extract_keywords(df)
    # print(job_title_keywords)
    frequency_of_keyword(df)

    # Applied the basic keyword matching to each resume
    df['Shortlisted'] = df.apply(lambda row: basic_keyword_matching(row['Resume'], job_title_keywords, row['Category']), axis=1)
    
    # Evaluate the model's performance
    total_resumes = len(df)
    shortlisted_resumes = df['Shortlisted'].sum()

    print(f"Total Resumes: {total_resumes}")
    print(f"Shortlisted Resumes: {shortlisted_resumes}")
    print(f"Shortlisting Rate: {shortlisted_resumes / total_resumes:.2%}")

    # Assuming 'Shortlisted' is the ground truth label
    y_true = df['Shortlisted']
    y_pred = df['Shortlisted']

    # Print classification report and accuracy
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    
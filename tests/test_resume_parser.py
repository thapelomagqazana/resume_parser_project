import unittest
import pandas as pd
from src.main import load_dataset, extract_keywords, frequency_of_keyword, basic_keyword_matching

class TestResumeParser(unittest.TestCase):

    def setUp(self):
        # Load the dataset for testing
        self.dataset_path = 'data/UpdatedResumeDataSet.csv'
        self.df = load_dataset(self.dataset_path)
        self.job_title_keywords = extract_keywords(self.df)

    def test_positive_case(self):
        # Positive case: A resume that should be shortlisted
        resume_text = "Experienced data scientist with expertise in machine learning."
        job_title = "Data Scientist"
        result = basic_keyword_matching(resume_text, self.job_title_keywords, job_title)
        self.assertTrue(result)

    def test_negative_case(self):
        # Negative case: A resume that should not be shortlisted
        resume_text = "Entry-level candidate with limited experience."
        job_title = "Data Scientist"
        result = basic_keyword_matching(resume_text, self.job_title_keywords, job_title)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()

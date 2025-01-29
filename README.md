# Job-Search-Recommendation-Tool---Indeed.com

##### Team Members:
Michael Crosson, Andy Ma, Destin Blanchard, Sarah Dominguez, Brooks Li, Brinda Asuri

### Project Overview
This project is designed to help job seekers efficiently find personalized job postings that align with their skills, experiences, and preferences. By leveraging **web scraping, Natural Language Processing (NLP), and Machine Learning (ML)** techniques, our system provides tailored job recommendations.

### Features
- **Web Scraping:** Uses Playwright and Beautiful Soup to extract job listings from Indeed.com.
- **Topic Analysis with NMF:** Extracts key topics from job descriptions and candidate preferences using Non-Negative Matrix Factorization (NMF) and TF-IDF.
- **Cosine Similarity Matching:** Computes similarity scores between a candidate’s resume, job preferences, and job descriptions.
- **Final Recommendation Ranking:** Assigns scores based on: Preference similarity, Job description similarity, Skill capture rate

### Tech Stack
- **Programming Languages:** Python
- **Libraries & Tools:** Playwright, Beautiful Soup, Pandas, NumPy, Scikit-Learn, TensorFlow, NLTK, TF-IDF
- **Data Processing:** TF-IDF for text vectorization, NMF for topic modeling
- **Similarity Matching:** Cosine similarity



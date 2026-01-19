# Coursera Course Recommender 🎓

A Machine Learning web application that recommends Coursera courses based on user skills, preferred difficulty, and specific goals.

### 🚀 Live Demo
https://course-recommendation-system-by-navya.streamlit.app/

### 📊 Dataset
Used the [Coursera Courses Metadata & Reviews 2025](https://www.kaggle.com/datasets/juliasevaslidou/coursera-courses-metadata-and-reviews-dataset-2025) from Kaggle.

### 🛠️ How it works
1. **Data Processing:** Cleaned and merged course metadata with user reviews.
2. **Natural Language Processing:** Used NLTK for stemming and Scikit-Learn's TFIDFVectorizer to convert course descriptions into vectors.
3. **Similarity Engine:** Uses Cosine Similarity to match user queries with the most relevant courses.
4. **Ranking:** Applied a custom ranking algorithm that boosts courses with higher ratings.

### 📂 Project Structure
- `app.py`: The Streamlit web interface logic.
- `Data_Cleaning.ipynb`: The original notebook used for data exploration and preprocessing.
- `course_list.pkl`: The processed data exported for the app.

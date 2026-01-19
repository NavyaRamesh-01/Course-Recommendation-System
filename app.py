import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# --- HELPER FUNCTIONS ---
def stemming(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

@st.cache_resource # This ensures the math only happens once
def load_and_process_data():
    df = pickle.load(open('course_list.pkl', 'rb'))
    cv = TfidfVectorizer(max_features=5000, stop_words='english',ngram_range=(1,2))
    vectors = cv.fit_transform(df['tags']).toarray()
    return df, vectors, cv

new_df, vectors, cv = load_and_process_data()

def recommend_courses(user_vector, new_df, user_difficulty, course_vectors):
    similarities = cosine_similarity(user_vector, course_vectors).flatten()
    sorted_indices = similarities.argsort()[::-1]
    
    recommendations = []
    min_rating = new_df['final_rating'].min()
    max_rating = new_df['final_rating'].max()
    rating_boosts = ((new_df['final_rating'] - min_rating) / (max_rating - min_rating)).fillna(0) + 0.5
    
    for idx in sorted_indices:
        course_difficulty = new_df.iloc[idx]['level'].lower()
        course_rating = new_df.iloc[idx]['final_rating']
        raw_similarity = similarities[idx]
        
        if user_difficulty != 'mixed' and course_difficulty != user_difficulty.lower():
            continue
        if course_rating < 3.5:
            continue
            
        final_score = (raw_similarity * 0.9) + (rating_boosts.iloc[idx] * 0.1)
        recommendations.append({
            'Title': new_df.iloc[idx]['final_title'],
            'Difficulty': course_difficulty.capitalize(),
            'Rating': course_rating,
            'URL': new_df.iloc[idx]['url'],
            'Score': final_score
        })
        if len(recommendations) >= 5:
            break
            
    return pd.DataFrame(recommendations)

# --- UI CODE ---
st.set_page_config(page_title="Course Recommender")
st.title("🎓 Course Recommender System")

user_skill = st.text_input("1. Which skill do you want to learn?", placeholder="e.g. Python")
user_difficulty = st.selectbox("2. What level?", ["Beginner", "Intermediate", "Mixed"])
user_description = st.text_area("3. Describe your goal", placeholder="e.g. project-based learning")

if st.button('Recommend Courses'):
    if user_skill:
        user_query = f"{user_skill} {user_skill} {user_skill} {user_description}"
        stemmed_query = stemming(user_query.lower())
        user_vector = cv.transform([stemmed_query]).toarray()
        
        res_df = recommend_courses(user_vector, new_df, user_difficulty, vectors)
        
        if not res_df.empty:
            for i, row in res_df.iterrows():
                st.write(f"### {row['Title']}")
                st.write(f"**Level:** {row['Difficulty']} | **Rating:** ⭐ {row['Rating']}")
                st.link_button("View Course", row['URL'])
                st.divider()
        else:
            st.error("No matches found. Try modifying the search!")

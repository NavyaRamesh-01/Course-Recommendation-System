import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# --- CONFIGURATION ---
st.set_page_config(page_title="Course Recommender AI", page_icon="🎓", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .course-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #7e57c2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .main-title {
        color: #7e57c2;
        font-size: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

ps = PorterStemmer()

def stemming(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

@st.cache_resource 
def load_and_process_data():
    df = pickle.load(open('course_list.pkl', 'rb'))
    cv = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
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
        
        if user_difficulty.lower() != 'mixed' and course_difficulty != user_difficulty.lower():
            continue
        if course_rating < 3.5:
            continue
            
        final_score = (raw_similarity * 0.9) + (rating_boosts.iloc[idx] * 0.1)
        recommendations.append({
            'Title': new_df.iloc[idx]['final_title'].title(),
            'Difficulty': course_difficulty.capitalize(),
            'Rating': course_rating,
            'URL': new_df.iloc[idx]['url']
        })
        if len(recommendations) >= 6: 
            break
            
    return pd.DataFrame(recommendations)

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Set your preferences")
    user_skill = st.text_input("🎯 What skill?", placeholder="e.g. SQL")
    user_difficulty = st.selectbox("📊 Level", ["Beginner", "Intermediate", "Mixed"])
    user_description = st.text_area("📝 Additional info", placeholder="e.g. project-based learning")
    
    st.markdown("---")
    predict_button = st.button('Search Courses', use_container_width=True)

# --- MAIN PAGE UI ---
if predict_button:
    if user_skill:
        with st.spinner('Finding the best courses...'):
            user_query = f"{user_skill} {user_skill} {user_skill} {user_description}"
            stemmed_query = stemming(user_query.lower())
            user_vector = cv.transform([stemmed_query]).toarray()
            
            res_df = recommend_courses(user_vector, new_df, user_difficulty, vectors)
            
            if not res_df.empty:
                st.toast('Results found!', icon='🎉')
                st.markdown(f"## Best matches for '{user_skill}'")
                
                col1, col2 = st.columns(2)
                for i, row in res_df.iterrows():
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"""
                            <div class="course-card">
                                <h3 style="color: #7e57c2; margin-bottom:0;">{row['Title']}</h3>
                                <p style="color: gray;"><b>{row['Difficulty']}</b> | ⭐ {row['Rating']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.link_button("View Course on Coursera", row['URL'], use_container_width=True)
                        st.write("") 
            else:
                st.error("No matches found. Try modifying your search!")
    else:
        st.warning("Please enter a skill in the sidebar.")

else:
    # --- WELCOME SCREEN WITH FEMALE ILLUSTRATION ---
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("https://illustrations.popsy.co/purple/studying.svg", width=500)
        st.markdown("<h1 style='text-align: center; color: #7e57c2;'>Ready to Learn?</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2em;'>Use the sidebar to search for courses. Our AI will find the perfect matches for your skills and experience level.</p>", unsafe_allow_html=True)

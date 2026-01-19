import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# --- CONFIGURATION ---
st.set_page_config(page_title="Course AI", page_icon="💡", layout="wide")

# Modern Aesthetic CSS
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }

    /* Course Card Styling */
    .course-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border-top: 4px solid #00b894; /* Aesthetic Emerald Green */
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    
    /* Typography */
    h1 {
        color: #2d3436;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
    }
    .course-title {
        color: #2d3436;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .difficulty-tag {
        background-color: #e8f8f5;
        color: #00b894;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2d3436;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b894;
        color: white;
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
    st.markdown("<br>", unsafe_allow_html=True)
    # Aesthetic Minimalist Icon
    st.image("https://storyset.com/blog/wp-content/uploads/2021/04/Education-bro.png", use_column_width=True)
    st.markdown("### Preferences")
    user_skill = st.text_input("What skill?", placeholder="e.g. Data Science")
    user_difficulty = st.selectbox("Level", ["Beginner", "Intermediate", "Mixed"])
    user_description = st.text_area("Learning Goal", placeholder="e.g. I want to build real projects")
    
    predict_button = st.button('Discover Courses', use_container_width=True)

# --- MAIN PAGE UI ---
if predict_button:
    if user_skill:
        with st.spinner('Thinking...'):
            user_query = f"{user_skill} {user_skill} {user_skill} {user_description}"
            stemmed_query = stemming(user_query.lower())
            user_vector = cv.transform([stemmed_query]).toarray()
            
            res_df = recommend_courses(user_vector, new_df, user_difficulty, vectors)
            
            if not res_df.empty:
                st.toast('Matches found!')
                st.markdown(f"## Courses curated for your journey")
                
                col1, col2 = st.columns(2)
                for i, row in res_df.iterrows():
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"""
                            <div class="course-card">
                                <div class="course-title">{row['Title']}</div>
                                <span class="difficulty-tag">{row['Difficulty']}</span>
                                <span style="margin-left: 10px; color: #636e72;">⭐ {row['Rating']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        st.link_button("Go to Course", row['URL'], use_container_width=True)
                        st.write("") 
            else:
                st.error("Try adjusting your keywords!")
    else:
        st.warning("Please enter a skill.")

else:
    # --- AESTHETIC WELCOME SCREEN ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.5, 2, 0.5])
    with c2:
        # Using a very aesthetic, soft-colored Storyset illustration
        st.image("https://storyset.com/illustration/creative-writing/amico/google-color", use_column_width=True)
        st.markdown("<h1 style='text-align: center;'>Expand your horizons.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #636e72; font-size: 1.1em;'>Our AI assistant analyzes thousands of data points to find the perfect learning path for your unique career goals.</p>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# --- CONFIGURATION ---
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .course-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6c63ff;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        color: #31333F;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

ps = PorterStemmer()

# --- HELPER FUNCTIONS (PRESERVING YOUR LOGIC) ---
def stemming(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return "".join(y)

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
            'URL': new_df.iloc[idx]['url'],
            'Score': final_score
        })
        if len(recommendations) >= 6: 
            break
            
    return pd.DataFrame(recommendations)

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/online-certification-concept_23-2148575662.jpg", use_column_width=True)
    st.title("Search Filters")
    user_skill = st.text_input("1. Skill to learn", placeholder="e.g. SQL")
    user_difficulty = st.selectbox("2. Difficulty", ["Beginner", "Intermediate", "Mixed"])
    user_description = st.text_area("3. Your Goal", placeholder="e.g. project-based learning")
    
    st.markdown("---")
    predict_button = st.button('Find Best Courses', use_container_width=True)

# --- MAIN PAGE UI ---
st.title("🎓 Course Recommender AI")
st.markdown("#### Discover top-rated Coursera courses tailored to your goals.")

if predict_button:
    if user_skill:
        with st.spinner('Curating your courses...'):
            user_query = f"{user_skill} {user_skill} {user_skill} {user_description}"
            stemmed_query = stemming(user_query.lower())
            user_vector = cv.transform([stemmed_query]).toarray()
            
            res_df = recommend_courses(user_vector, new_df, user_difficulty, vectors)
            
            if not res_df.empty:
                # Replacement for balloons: A subtle "Toast" notification
                st.toast('Recommendations ready!', icon='✅')
                
                st.success(f"We found {len(res_df)} courses matching your request:")
                
                col1, col2 = st.columns(2)
                
                for i, row in res_df.iterrows():
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"""
                            <div class="course-card">
                                <h3 style="color: #6c63ff;">{row['Title']}</h3>
                                <p><b>Level:</b> {row['Difficulty']} | ⭐ <b>Rating:</b> {row['Rating']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.link_button("🚀 Start Learning", row['URL'], use_container_width=True)
                        st.write("") 
            else:
                st.error("We couldn't find a perfect match. Try broadening your keywords!")
    else:
        st.warning("Please enter a skill in the sidebar.")
else:
    # MAIN SCREEN IMAGE (Female Learner)
    st.write("")
    col_img, col_txt = st.columns([1, 1])
    with col_img:
        st.image("https://img.freepik.com/free-vector/female-student-with-laptop-studying-online-at-home_23-2148530353.jpg", use_column_width=True)
    with col_txt:
        st.markdown("""
            <br><br><br>
            <h3>Ready to start your next chapter?</h3>
            <p style='font-size: 1.2em; color: gray;'>
                Use the sidebar to tell us what you want to learn. 
                Our AI analyzes thousands of Coursera courses to find the 
                perfect balance between relevance and high user ratings.
            </p>
        """, unsafe_allow_html=True)

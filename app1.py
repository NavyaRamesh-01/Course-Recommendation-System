import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# --- CONFIGURATION ---
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# Custom CSS for a better look
st.markdown("""
    <style>
    .course-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

ps = PorterStemmer()

# --- HELPER FUNCTIONS (KEEPING YOUR LOGIC) ---
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
            'URL': new_df.iloc[idx]['url'],
            'Score': final_score
        })
        if len(recommendations) >= 6: # Increased to 6 for better layout
            break
            
    return pd.DataFrame(recommendations)

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Settings")
    st.markdown("Enter your preferences below to find courses.")
    user_skill = st.text_input("1. Which skill?", placeholder="e.g. Python")
    user_difficulty = st.selectbox("2. What level?", ["Beginner", "Intermediate", "Mixed"])
    user_description = st.text_area("3. Describe your goal", placeholder="e.g. project-based learning")
    
    st.markdown("---")
    predict_button = st.button('Recommend Courses', use_container_width=True)

# --- MAIN PAGE UI ---
st.title("🎓 Course Recommender System")
st.markdown("#### Personalized learning paths driven by AI")

if predict_button:
    if user_skill:
        with st.spinner('Analyzing courses...'):
            user_query = f"{user_skill} {user_skill} {user_skill} {user_description}"
            stemmed_query = stemming(user_query.lower())
            user_vector = cv.transform([stemmed_query]).toarray()
            
            res_df = recommend_courses(user_vector, new_df, user_difficulty, vectors)
            
            if not res_df.empty:
                st.success(f"Found {len(res_df)} great matches!")
                st.balloons()
                
                # Display in a grid
                col1, col2 = st.columns(2)
                
                for i, row in res_df.iterrows():
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"""
                            <div class="course-card">
                                <h3>{row['Title']}</h3>
                                <p>📊 <b>Level:</b> {row['Difficulty']} | ⭐ <b>Rating:</b> {row['Rating']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.link_button("View Course Details", row['URL'], use_container_width=True)
                        st.write("") # Spacer
            else:
                st.error("No matches found. Try changing the keywords or level!")
    else:
        st.warning("Please enter a skill in the sidebar to get started.")
else:
    # Display this when the app first loads
    st.info("👈 Fill in your details in the sidebar and click 'Recommend Courses' to see results!")
    st.image("https://img.freepik.com/free-vector/learning-concept-illustration_114360-6186.jpg", width=600)

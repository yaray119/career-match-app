import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
from PyPDF2 import PdfReader

@st.cache_data
def load_jobs():
    df = pd.read_csv("clean_jobs.csv")
    df = df[['title', 'company', 'location', 'description']].dropna().reset_index(drop=True)
    tech_skills_list = [
        "python", "sql", "excel", "tableau", "r", "sas", "java", "c++", "c#", "javascript",
        "matlab", "spark", "hadoop", "tensorflow", "pytorch", "aws", "azure", "linux",
        "git", "bash", "docker", "kubernetes", "machine learning", "deep learning",
        "data analysis", "data visualization", "data science", "statistics", "etl", "nosql"
    ]
    def extract_skills(text, skills): return ", ".join([s for s in skills if s in text.lower()])
    df['tech_skills'] = df['description'].apply(lambda x: extract_skills(x, tech_skills_list))
    return df

def clean_text(text):
    return re.sub(r'[^a-zA-Z\\s]', '', text.lower())

def extract_text_from_resume(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        return docx2txt.process(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8")

def match_jobs(jobs_df, user_skills):
    user_text = clean_text(" ".join(user_skills))
    jobs_df['text'] = (jobs_df['title'] + " " + jobs_df['description']).apply(clean_text)
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(jobs_df['text'].tolist() + [user_text])
    similarities = cosine_similarity(vectors[:-1], vectors[-1])
    jobs_df['match_score'] = similarities
    user_set = set([s.strip().lower() for s in user_skills])
    def missing(techs):
        if not isinstance(techs, str): return []
        return list(set(techs.lower().split(", ")) - user_set)
    jobs_df['missing_skills'] = jobs_df['tech_skills'].apply(missing)
    return jobs_df.sort_values(by='match_score', ascending=False).head(10)

# Streamlit UI
st.title("CareerMatch: Personalized Job Fit Explorer")
st.markdown("Upload a resume or enter your skills to see your top job matches and missing skills.")

option = st.radio("Choose input method:", ["Upload Resume", "Manual Skill Entry"])
user_skills = []

if option == "Manual Skill Entry":
    skill_input = st.text_input("Enter your skills (comma-separated):", "python, data analysis, polymer")
    user_skills = [s.strip() for s in skill_input.lower().split(",") if s.strip()]
else:
    resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if resume_file:
        resume_text = extract_text_from_resume(resume_file)
        user_skills = [w for w in clean_text(resume_text).split() if len(w) > 3]
        st.success("Resume processed successfully!")

if user_skills:
    df = load_jobs()
    matched_jobs = match_jobs(df.copy(), user_skills)

    st.subheader("🔍 Top Matching Jobs")
    for _, row in matched_jobs.iterrows():
        st.markdown(f"**{row['title']}** at **{row['company']}** – *{row['location']}*")
        st.progress(float(row['match_score']))
        st.markdown(f"**Matching Skills:** `{', '.join(set(row['tech_skills'].split(', ')) & set(user_skills))}`")
        st.markdown(f"**Missing Skills:** `{', '.join(row['missing_skills']) if row['missing_skills'] else 'None ✅'}`")
        st.markdown("---")


This is your README. READMEs are where you can communicate what your project is and how to use it.

Write your name on line 6, save it, and then head back to GitHub Desktop.

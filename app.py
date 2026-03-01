import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(
    page_title= "DS Job Market Analysis",
    page_icon = "📊",
    layout = "wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/naukri_nlp.csv")
    salary_df = pd.read_csv("data/ds_salaries.csv")
    return df, salary_df

df, salary_df = load_data()

# Loading Data and Model
@st.cache_resource
def load_model():
    import json
    
    with open("data/feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    
    with open("data/encoders.json", "r") as f:
        encoder_dict = json.load(f)
    
    train_data = pd.read_csv("data/train_data.csv")
    
    return encoder_dict, feature_cols, train_data

encoder_dict, feature_cols, train_data = load_model()


# Sidebar
st.sidebar.title("DS Job Market")    
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ['Job Market Overview',
     'Skills Analysis',
     'Salary Predictor']
)

st.sidebar.markdown('---')
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"**{len(df)}** job postings" )
st.sidebar.markdown(f"**{df['city'].nunique()}** cities")
st.sidebar.markdown(f"**{df['company'].nunique()}** companies")
st.sidebar.markdown('---')
st.sidebar.markdown('Built by Swayam')


# Page 1 = Job market Overview

if page == 'Job Market Overview':
    
    st.title('Data Science Job Market - India')
    st.markdown('*Insights from Live Job postings on Naukri.com*')
    st.markdown('---')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs",len(df))
    with col2:
        st.metric("Company Hiring",df['company'].nunique())
    with col3:
        st.metric("Cities Covered",df['city'].nunique())
    with col4:
        remote_pct = round(
            (df['work_mode'] == 'Remote').sum() / len(df) * 100,1
        )
        st.metric('Remote Friendly',f'{remote_pct}%')
        
    st.markdown('---')
        
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader('Jobs by Role')
        role_counts = df['role'].value_counts().reset_index()
        role_counts.columns = ['role','count']
            
        fig = px.bar(
            role_counts,
            x='role',y='count',
            color = 'role',
            color_discrete_sequence = ["#1B4F72", "#1E8449", "#784212"],
            text = 'count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend = False,
            xaxis_title = '',
            yaxis_title = 'Job Postings'
        )
        st.plotly_chart(fig, use_container_width=True)
            
            
    with col2:
        st.subheader('Work Mode Distribution')
        work_counts = df['work_mode'].value_counts().reset_index()
        work_counts.columns = ['mode','count']
            
        fig = px.pie(
            work_counts,
            names = 'mode', values='count',
            color_discrete_sequence=["#1B4F72", "#2E86C1", "#AED6F1"],
            hole = 0.4
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig, use_container_width=True)
            
            
    st.subheader('Top Hiring Cities')
    city_counts = df['city'].value_counts().head(12).reset_index()
    city_counts.columns = ['city','count']
    
    fig = px.bar(
        city_counts,
        x='count', y='city',
        orientation='h',
        color='count',
        color_continuous_scale='Blues',
        text = 'count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        coloraxis_showscale=False,
        xaxis_title='Number of Job Postings',
        yaxis_title=''
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.subheader('Experience Required by Role')
    exp_data = df.dropna(subset=['exp_avg'])
    
    fig = px.histogram(
        exp_data,
        x='exp_avg',
        color='role',
        barmode='overlay',
        nbins=15,
        color_discrete_sequence=["#1B4F72", "#1E8449", "#784212"],
        opacity=0.75
    )
    fig.update_layout(
        xaxis_title='Average years of Experience Required',
        yaxis_title='Number of Jobs',
        legend_title='Role'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Page 2 Skills analysis
    
elif page == 'Skills Analysis':
    
    st.title('Skills Demand Analysis')
    st.markdown('*What skills are Indian Companies actually asking for?*')
    st.markdown('---')

    all_skills=[]
    for skills_str in df['skills'].dropna():
        skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
        all_skills.extend(skills)
        
    st.subheader('Filter by Role')
    selected_roles = st.multiselect(
        "Select roles to include",
        options = df['role'].unique().tolist(),
        default = df['role'].unique().tolist()
    )
    
    if not selected_roles:
        st.warning('Please select at least one role.')
        st.stop()
        
    filtered_df = df[df['role'].isin(selected_roles)]
    st.markdown(f"Showing **{len(filtered_df)}** job postings")
    st.markdown('---')
    
    st.subheader('Top 20 In-Demand Skills')
    
    filtered_skills = []
    for skills_str in filtered_df['skills'].dropna():
        skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
        filtered_skills.extend(skills)
    

    skill_counts = Counter(filtered_skills).most_common(20)
    filtered_freq = pd.DataFrame(skill_counts, columns=["skill", "count"])


    filtered_freq["percentage"] = (
        (filtered_freq["count"] / len(filtered_df)) * 100
    ).round(1)

    
    fig = px.bar(
        filtered_freq,
        x='percentage', y='skill',
        orientation = 'h',
        color='percentage',
        color_continuous_scale = 'Blues',
        text = 'percentage'
    )
    fig.update_traces(
        texttemplate="%{text}%",
        textposition='outside'
    )
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        xaxis_title = " % of Job Postings Mentioning this skills",
        yaxis_title = '',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig,use_container_width=True)
    
    
    st.subheader('Top skills by role')
    
    role_cols = st.columns(len(selected_roles))
    
    role_colors = {
        "Data Scientist": "#1B4F72",
        "ML Engineer": "#1E8449",
        "Data Analyst": "#784212"
    }
    
    for idx, role in enumerate(selected_roles):
        role_skills = []
        for skills_str in filtered_df[filtered_df['role'] == role]['skills'].dropna():
            skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
            role_skills.extend(skills)
            
        top_role = pd.DataFrame(
            Counter(role_skills).most_common(8),
            columns=['skill','count']
        )
        
        color = role_colors.get(role, "#1B4F72")
        
        
        fig = px.bar(
            top_role,
            x='count', y='skill',
            orientation = 'h',
            title=role,
            color_discrete_sequence=[color]
        )
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            showlegend = False,
            xaxis_title = 'Count',
            yaxis_title=''
        )
        role_cols[idx].plotly_chart(fig, use_container_width = True)
        
        
    # Wordcloud
    
    st.subheader("Skills Word Cloud")
    st.markdown("*Larger words appear in more job postings*")
    
    
    skill_text = ' '.join([
        skill.replace(' ','_')
        for skill, count in Counter(filtered_skills).items()
        for _ in range(count)
    ])
    
    wordcloud = WordCloud(
        width = 1400,
        height=500,
        background_color = 'white',
        colormap = 'Blues',
        max_words=80,
        prefer_horizontal=0.9
    ).generate(skill_text)
    
    fig_wc, ax=plt.subplots(figsize=(16,8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)
    plt.close(fig_wc)
    
    
# Page 3 is Salary Predictor


elif page == 'Salary Predictor':
    
    st.title("Data Science Salary Predictor")
    st.markdown("*Predict Salary based on your profile*")
    st.markdown("---")
    
    st.info(
        "Model Trained on Global Data Science salary data."
        "Predictions are in USD. Average prediction error is ~$39,000."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        experience_level = st.selectbox(
            'Experience Level',
            ['EN – Entry Level', 'MI – Mid Level',
             'SE – Senior Level','EX – Executive Level']
        )
        
        employment_type = st.selectbox(
            "Employment Type",
            ['FT – Full Time', 'PT – Part Time',
             'CT – Contract', 'FL – Freelance'
             ]
        )
        
        company_size = st.selectbox(
            "Company Size",
            ['S – Small', 'M – Medium', 'L – Large']
        )
        
        
    with col2:
        remote_ratio = st.selectbox(
            "Work Mode",
            ['0 – On-site', '50 – hybrid', '100 – Remote']
        )
        
        work_year = st.selectbox(
            'Year',
            [2023, 2022, 2021, 2000]
        )
        
        employee_residence = st.selectbox(
            "Your Location (Country Code)",
            sorted(salary_df['employee_residence'].unique().tolist())
        )
        
        company_location = st.selectbox(
            "Company Location (Country Code)",
            sorted(salary_df['company_location'].unique().tolist())
        )
        
         
    st.markdown("---")
    
    predict_btn = st.button("Predict Salary", type="primary")
    
    if predict_btn:
        try:
            # Parse codes
            exp_code = experience_level.split(" – ")[0]
            emp_code = employment_type.split(" – ")[0]
            size_code = company_size.split(" – ")[0]
            remote_code = int(remote_ratio.split(" – ")[0])

            # Encode input using saved encoder dictionaries
            def encode_value(col, value):
                if col in encoder_dict and value in encoder_dict[col]:
                    return encoder_dict[col][value]
                return 0

            encoded_input = {
                "work_year": work_year,
                "experience_level": encode_value("experience_level", exp_code),
                "employment_type": encode_value("employment_type", emp_code),
                "employee_residence": encode_value("employee_residence", employee_residence),
                "remote_ratio": remote_code,
                "company_location": encode_value("company_location", company_location),
                "company_size": encode_value("company_size", size_code),
                "job_category": encode_value("job_category", "data_scientist"),
                "is_us_employee": int(employee_residence == "US"),
                "is_us_company": int(company_location == "US")
            }

            # Find similar profiles in training data
            # This is a nearest neighbor approach — find closest match
            input_df = pd.DataFrame([encoded_input])[feature_cols]
            
            # Calculate similarity score for each training row
            train_features = train_data[feature_cols]
            
            # Weight experience level heavily as it's strongest predictor
            weights = {col: 1 for col in feature_cols}
            weights["experience_level"] = 3
            weights["is_us_employee"] = 3
            weights["is_us_company"] = 2

            distances = pd.Series(0.0, index=train_data.index)
            for col in feature_cols:
                weight = weights.get(col, 1)
                distances += weight * (train_features[col] - input_df[col].values[0]) ** 2

            # Get 20 most similar profiles
            closest_idx = distances.nsmallest(20).index
            similar_salaries = train_data.loc[closest_idx, "salary"]
            prediction = similar_salaries.median()

            # Display results
            st.success(f"### Predicted Annual Salary: ${prediction:,.0f} USD")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual (USD)", f"${prediction:,.0f}")
            with col2:
                st.metric("Monthly (USD)", f"${prediction/12:,.0f}")
            with col3:
                inr = prediction * 83
                st.metric("Annual (INR approx)", f"₹{inr:,.0f}")

            st.markdown("---")
            st.subheader("How Does This Compare?")

            exp_median = salary_df[
                salary_df["experience_level"] == exp_code
            ]["salary_in_usd"].median()

            overall_median = salary_df["salary_in_usd"].median()

            comparison_df = pd.DataFrame({
                "category": ["Your Prediction", 
                            f"{exp_code} Level Median", 
                            "Overall Median"],
                "salary": [prediction, exp_median, overall_median]
            })

            fig = px.bar(
                comparison_df,
                x="category", y="salary",
                color="category",
                color_discrete_sequence=["#1B4F72", "#2E86C1", "#AED6F1"],
                text="salary"
            )
            fig.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside"
            )
            fig.update_layout(
                showlegend=False,
                yaxis_title="Salary (USD)",
                xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "⚠️ Prediction uses nearest neighbor matching on training data. "
                "Use as directional guidance only."
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
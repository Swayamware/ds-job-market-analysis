# 📊 Data Science Job Market Analysis — India

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ds-job-market-analysis-m5s3lff8bubzn2appja9xyc.streamlit.app/)

A data science project that scrapes, analyzes, and visualizes the Indian data science job market using real job postings from Naukri.com — covering roles across Data Scientist, ML Engineer, and Data Analyst positions.

**Built by Swayam Ware**

---

## 🔗 Live Demo

👉 [Open the App](https://ds-job-market-analysis-m5s3lff8bubzn2appja9xyc.streamlit.app/)

---

## 📌 Project Overview

This project answers two questions that matter to anyone entering the data science field:

1. **What does the Indian data science job market actually look like?** — Which skills are in demand, which cities are hiring, and what experience levels are companies looking for?
2. **What factors influence data science salaries globally?** — Using a curated global dataset, what can we predict about salary given experience, location, and company profile?

The project is built end-to-end — from raw data collection through cleaning, analysis, modeling, and deployment as an interactive web application.

---

## 🗂️ Project Structure

```
ds-job-market-analysis/
│
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── naukri_raw.csv            # Raw scraped data
│   ├── naukri_clean.csv          # Cleaned data
│   ├── naukri_nlp.csv            # Enriched data with extracted skills
│   ├── ds_salaries.csv           # Kaggle salary dataset
│   ├── encoders.json             # Label encoders for ML model
│   ├── feature_columns.json      # Model feature order
│   └── train_data.csv            # Training data for salary prediction
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Collection | Python, Selenium, BeautifulSoup |
| Data Processing | Pandas, NumPy, Regex |
| Visualization | Plotly, Matplotlib, Seaborn, WordCloud |
| Machine Learning | Scikit-learn (Gradient Boosting, Random Forest) |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🔍 Key Findings

**Skills Demand (Indian Market)**
- Machine Learning and Python are the most demanded skills, each appearing in nearly 48% of all job postings — indicating they are non-negotiable for most data roles in India.
- SQL appears in approximately 19% of postings, suggesting that while essential, it is often assumed rather than explicitly listed.
- Bengaluru dominates hiring with 27 out of every 100 data role postings, followed by Pune and Hyderabad.

**Work Mode Trends**
- The majority of Indian data science roles remain office-based or hybrid. Fully remote positions represent a small fraction of the market, contrasting with global trends.

**Experience Requirements**
- Most data scientist roles require 2–5 years of experience, making it competitive for freshers. Data analyst roles show a lower experience bar and serve as a more accessible entry point.

**Salary Prediction (Global Dataset)**
- Experience level is the strongest predictor of salary, followed by geographic location.
- A US-based senior data scientist earns significantly more than the global median, which creates high variance in global salary models.
- The Gradient Boosting model achieved an R² of 0.353 on the global dataset — a result that reflects the inherent difficulty of salary prediction across geographies rather than model weakness. Key salary drivers such as city-level location, company name, and industry were not available in the dataset.

---

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/Swayamware/ds-job-market-analysis.git
cd ds-job-market-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📈 Dashboard Pages

**Page 1 — Job Market Overview**
Displays key metrics, job distribution by role, work mode breakdown, top hiring cities, and experience requirements — all derived from live-scraped Naukri.com data.

**Page 2 — Skills Analysis**
Interactive skills demand chart filterable by role, top skills comparison across Data Scientist, ML Engineer, and Data Analyst roles, and a skills word cloud showing relative demand visually.

**Page 3 — Salary Predictor**
Input your experience level, employment type, company size, work mode, and location to receive a salary estimate with comparison against role-level and overall medians.

---

## 🔮 Future Improvements

- **Automate data refresh** — Schedule weekly scraping to keep job market insights current rather than static.
- **Expand data sources** — Incorporate LinkedIn and Indeed data to increase dataset size and geographic coverage.
- **Improve salary model** — Source a richer dataset with city-level granularity, company names, and exact years of experience to meaningfully improve prediction accuracy.
- **Add job recommendation** — Allow users to input their skill profile and receive ranked job matches from the scraped dataset using TF-IDF cosine similarity.
- **India-specific salary data** — The current salary model is trained on global data. An India-specific salary dataset would make the predictor far more relevant for the target audience.

---

## 📬 Contact

**Swayam Ware**
- GitHub: [@Swayamware](https://github.com/Swayamware)
- Linkedin: (www.linkedin.com/in/swayamware)
---

*Data scraped from Naukri.com. Salary data sourced from Kaggle. This project is intended for educational and portfolio purposes.*

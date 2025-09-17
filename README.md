# LinkedIn Job Postings Analysis (2023-2024)

## Overview
This project explores a rich dataset of LinkedIn job postings covering the years 2023 and 2024, comprising over 280,000 job records with 35 features. The analysis aims to uncover insights into job market dynamics, skill demand, geographic distributions, compensation trends, and employer profiles.

## Key Areas of Analysis
- **Job Market Dynamics:** Trends in job posting volume by location, industry, and company size to identify talent demand hotspots.
- **Work Type and Flexibility:** Patterns in remote work availability and types of employment offered.
- **Skills Demand:** Examination of popular skills and emerging competencies asked by employers.
- **Geographic and Market Insights:** Distribution of job postings by city, state, and country, alongside job popularity metrics.
- **Compensation and Application Metrics:** Analysis of salary ranges, pay periods, and application counts.
- **Employer Profiles:** Insights into top recruiting companies and job posting platforms.

## Notebook Contents
1. **Data Import and Inspection:** Load data, check dataset structure, column data types, missing values, and duplicates.
2. **Data Cleaning:** Handle missing values via median, mode imputation, and forward/backward filling; drop columns with excessive null data; cap numeric outliers.
3. **Exploratory Data Analysis (EDA):**
   - Univariate analysis of categorical and numerical features.
   - Bivariate visualizations exploring relationships between work types, experience levels, locations, and skills.
4. **Geographic Insights:** Detailed analysis of job distribution, average job view counts across locations.
5. **Visualizations:** Word clouds of job titles, skills, and industries; bar charts for top companies, posting domains, and job types.
6. **Feature Engineering:** Location parsing into city, state, and country fields for granular insights.
7. **Advanced Plots:** Interactive Plotly visualizations for multi-dimensional insights including stacked bar charts and grouped plots.

## Tools and Libraries
- Python (3.x)
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for basic visualization
- Plotly for interactive charts and advanced visuals
- WordCloud for creating word cloud visualizations

## Usage Instructions
- Ensure all required libraries are installed (use `pip install` for missing packages).
- Load the dataset (`mergedoutput.csv` or equivalent) before running the analysis cells.
- Follow the notebook cell order for a clean flow: Data loading  Cleaning  EDA  Visualization.
- Interactive Plotly graphs embed in the notebook; export options include converting to HTML or using screenshots for sharing visuals.

## Insights and Applications
- Helps job seekers identify trending roles and required skills.
- Assists employers and recruiters in benchmarking market demand.
- Provides policymakers with labor market supply-demand data.
- Can be extended to predictive modeling or integrated into job recommendation systems.

## Author
Ratnesh Kr. Satyarthi  
Data Analyst & Machine Learning Practitioner

## License
This project is open-source under the MIT License.

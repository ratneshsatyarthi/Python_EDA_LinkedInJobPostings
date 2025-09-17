from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
import io
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

app = Flask(__name__)
DATA_PATH = 'data/cleaned_linkedin_job_postings.csv'

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop columns with >60% nulls
    null_percents = df.isnull().mean() * 100
    drop_cols = null_percents[null_percents > 60].index.tolist()
    df = df.drop(columns=drop_cols)
    # Fill numeric nulls with median
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill categorical nulls with mode or 'Unknown'
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    return df

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')     

def generate_wordcloud(text, colormap='Set1'):
    wordcloud = WordCloud(width=1600, height=800, background_color='white', colormap=colormap).generate(text)
    fig = plt.figure(figsize=(16,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    return fig_to_base64(fig)

def generate_interactive_boxplot_html(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    n = len(numeric_cols)
    cols = 4
    rows = (n // cols) + (n % cols > 0)

    fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=numeric_cols)

    for i, col in enumerate(numeric_cols):
        row = (i // cols) + 1
        col_num = (i % cols) + 1
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxmean='sd',
                boxpoints='outliers',
                orientation='v'
            ),
            row=row,
            col=col_num
        )

    fig.update_layout(
        height=300 * rows,
        width=1200,
        #title_text="Boxplots of Numeric Columns (Outlier Removed)",
        showlegend=False
    )
    
    # Return interactive HTML div string (no full page, just the plot)
    return pio.to_html(fig, full_html=False)
        
def generate_histograms(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    histograms = []
    for col in numeric_cols:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Histogram + KDE on top subplot
        sns.histplot(df[col], bins=30, kde=True, color='red', ax=ax1)
        ax1.set_title(f'Histogram & Boxplot of {col}')
        ax1.set_ylabel('Frequency')
        ax1.grid(True)
        
        # Boxplot on bottom subplot, horizontal orientation by default
        sns.boxplot(x=df[col], ax=ax2, color='lightgray', fliersize=3)
        ax2.set_xlabel(col)
        ax2.set_yticks([])
        ax2.grid(False)
        
        plt.tight_layout()
        
        img = fig_to_base64(fig)
        histograms.append({'col': col, 'img_data': img})
    return histograms

def generate_top10_categorical_barplots(df):
    """
    Generate interactive Plotly horizontal bar plots for top 10 categories
    of specified categorical columns with subplot titles.
    Returns HTML div string for rendering in Flask templates.
    """
    cols = ['formatted_experience_level', 'posting_domain', 'company_name', 'skill_name', 'industry_name']
    custom_titles = [
        'Experience Level Distribution',
        'Top 10 Job Posting Domains',
        'Top 10 Companies',
        'Top 10 Key Skills in Demand',
        'Top 10 Industries'
    ]

    fig = make_subplots(
        rows=len(cols),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=custom_titles
    )

    for i, col in enumerate(cols, 1):
        count_data = df[col].value_counts().sort_values(ascending=False).head(10).reset_index()
        count_data.columns = [col, 'count']
        count_data = count_data.iloc[::-1]  # Reverse order so highest count on top

        fig.add_trace(
            go.Bar(y=count_data[col], x=count_data['count'], orientation='h', name=col),
            row=i,
            col=1
        )

    # Bold and center subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, color='black', family='Arial', weight='bold')
        annotation['x'] = 0.5
        annotation['xanchor'] = 'center'
        annotation['yanchor'] = 'bottom'

    fig.update_layout(
        height=300 * len(cols),
        showlegend=False
    )

    # Return HTML string of the Plotly figure to embed in template with safe rendering
    import plotly.io as pio
    return pio.to_html(fig, full_html=False)

def generate_categorical_distribution_bars(df):
    """
    Generates interactive Plotly bar charts for key categorical features:
    'formatted_experience_level', 'application_type', and 'formatted_work_type',
    arranged as 1 row and 3 columns subplot figure.
    Returns the HTML div string for embedding in Flask templates.
    """
    # Compute value counts sorted by index for consistent category ordering
    exp_levels = df['formatted_experience_level'].value_counts().sort_index()
    app_types = df['application_type'].value_counts().sort_index()
    work_types = df['formatted_work_type'].value_counts().sort_index()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('<b>Experience Level</b>',
                        '<b>Application Type</b>',
                        '<b>Work Type</b>')
    )

    # Add bar traces
    fig.add_trace(
        go.Bar(x=exp_levels.index, y=exp_levels.values, marker_color='blue', name='Jobs'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=app_types.index, y=app_types.values, marker_color='green', name='Jobs'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=work_types.index, y=work_types.values, marker_color='orange', name='Jobs'),
        row=1, col=3
    )

    # Update layout and axes
    fig.update_layout(height=500, width=1100, showlegend=False)
    for i in range(1, 4):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), row=1, col=i)
    fig.update_yaxes(title_text='Number of Jobs', row=1, col=1)

    # Return as HTML string (without full HTML wrapper) for embedding
    return pio.to_html(fig, full_html=False)

def generate_worktype_explevel_distribution(df):
    """
    Generates an interactive Plotly grouped histogram showing the distribution
    of jobs by 'formatted_work_type' and colored by 'formatted_experience_level'.
    Y-axis is log-scaled to accommodate varying counts.
    Returns an HTML div string for embedding in Flask templates.
    """
    fig = px.histogram(
        df,
        x='formatted_work_type',
        color='formatted_experience_level',
        category_orders={
            'formatted_work_type': df['formatted_work_type'].value_counts().index.tolist()
        },
        barmode='group',
        labels={
            'formatted_work_type': 'Work Type',
            'count': 'Number of Jobs',
            'formatted_experience_level': 'Experience Level'
        },
        #title='Distribution of Jobs by Work Type and Experience Level',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        title=dict(
            #text='<b>Distribution of Jobs by Work Type and Experience Level (Log Scale)</b>',
            x=0.5,
            xanchor='center'
        ),
        yaxis_title='<b>Number of Jobs</b>',
        showlegend=True
    )
    
    fig.update_yaxes(type='log')
    
    return pio.to_html(fig, full_html=False)

def generate_explevel_worktype_distribution(df):
    """
    Generates an interactive Plotly grouped histogram showing the distribution
    of jobs by 'formatted_experience_level' and colored by 'formatted_work_type'.
    Y-axis is log-scaled.
    Returns HTML div string for embedding in Flask templates.
    """
    fig = px.histogram(
        df,
        x='formatted_experience_level',
        color='formatted_work_type',
        category_orders={
            'formatted_experience_level': df['formatted_experience_level'].value_counts().index.tolist()
        },
        barmode='group',
        labels={
            'formatted_experience_level': 'Experience Level',
            'count': 'Number of Jobs',
            'formatted_work_type': 'Work Type'
        },
        #title='Distribution of Jobs by Experience Level and Work Type',
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        title=dict(
            #text='<b>Distribution of Jobs by Experience Level and Work Type (Log Scale)</b>',
            x=0.5,
            xanchor='center'
        ),
        yaxis_title='<b>Number of Jobs</b>',
        showlegend=True
    )
    
    fig.update_yaxes(type='log')
    
    return pio.to_html(fig, full_html=False)

def generate_top_locations_barplots(df):
    """
    Generates interactive Plotly bar charts showing top 10 Cities, States, and Countries
    by number of job postings, arranged as subplots in one row with three columns.
    Returns HTML div string for embedding in Flask templates.
    """
    city_counts = df['city'].value_counts().nlargest(10)
    state_counts = df['state'].value_counts().nlargest(10)
    country_counts = df['country'].value_counts().nlargest(10)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['<b>Top 10 Cities</b>', '<b>Top 10 States</b>', '<b>Top 10 Countries</b>']
    )

    fig.add_trace(
        go.Bar(x=city_counts.index, y=city_counts.values, marker_color='crimson', name='Job Postings'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=state_counts.index, y=state_counts.values, marker_color='coral', name='Job Postings'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=country_counts.index, y=country_counts.values, marker_color='forestgreen', name='Job Postings'),
        row=1, col=3
    )

    fig.update_layout(
        height=500, width=1200,
        showlegend=False,
        # title_text='Top 10 Cities, States, and Countries by Job Postings'
    )

    for i in range(1, 4):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), row=1, col=i)
        fig.update_yaxes(title_text='Job Postings', row=1, col=i)

    return pio.to_html(fig, full_html=False)

def generate_worktype_distribution_stacked(df):
    """
    Generates interactive stacked bar charts showing Work Type distribution among
    top 10 Cities and States with legend on Cities subplot only.
    Returns HTML div string suitable for embedding in Flask templates.
    """
    # Get top 10 cities and states by job count
    top_cities = df['city'].value_counts().nlargest(10).index.tolist()
    top_states = df['state'].value_counts().nlargest(10).index.tolist()

    # Filter data for top locations
    df_cities = df[df['city'].isin(top_cities)]
    df_states = df[df['state'].isin(top_states)]

    # Group by location and work type
    city_worktype = df_cities.groupby(['city', 'formatted_work_type']).size().unstack(fill_value=0)
    state_worktype = df_states.groupby(['state', 'formatted_work_type']).size().unstack(fill_value=0)
    work_types = city_worktype.columns.tolist()

    # Color palette and mapping
    palette = px.colors.qualitative.Bold
    color_map = {wt: palette[i % len(palette)] for i, wt in enumerate(work_types)}

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Top 10 Cities</b>', '<b>Top 10 States</b>')
    )

    # Add stacked bars for cities (show legend)
    for wt in work_types:
        fig.add_trace(go.Bar(
            x=city_worktype.index,
            y=city_worktype[wt],
            name=wt,
            legendgroup=wt,
            marker_color=color_map[wt],
            showlegend=True,
            marker_line_width=0
        ), row=1, col=1)

    # Add stacked bars for states (hide legend)
    for wt in work_types:
        fig.add_trace(go.Bar(
            x=state_worktype.index,
            y=state_worktype[wt],
            name=wt,
            legendgroup=wt,
            marker_color=color_map[wt],
            showlegend=False,
            marker_line_width=0
        ), row=1, col=2)

    # Update layout and axes
    fig.update_layout(
        barmode='stack',
        height=600, width=1100,
        showlegend=True,
        #title_text='<b>Work Type Distribution</b>',
        title_x=0.5
    )
    for i in range(1, 3):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), row=1, col=i)
        fig.update_yaxes(title_text='Job Postings', row=1, col=i)

    # Return HTML div string for embedding
    return pio.to_html(fig, full_html=False)

def generate_workexperience_distribution_stacked(df):
    """
    Generates interactive stacked bar charts showing work experience level distribution
    among top 10 cities and states.
    Returns HTML div string for embedding in Flask templates.
    """
    # Get top 10 cities and states by job count
    top_cities = df['city'].value_counts().nlargest(10).index.tolist()
    top_states = df['state'].value_counts().nlargest(10).index.tolist()

    # Filter dataframe for those locations
    df_cities = df[df['city'].isin(top_cities)]
    df_states = df[df['state'].isin(top_states)]

    # Group by experience level
    city_exp = df_cities.groupby(['city', 'formatted_experience_level']).size().unstack(fill_value=0)
    state_exp = df_states.groupby(['state', 'formatted_experience_level']).size().unstack(fill_value=0)

    experience_levels = city_exp.columns.tolist()

    # Color palette and mapping
    palette = px.colors.qualitative.T10
    color_map = {exp: palette[i % len(palette)] for i, exp in enumerate(experience_levels)}

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Top 10 Cities</b>', '<b>Top 10 States</b>')
    )

    # Add stacked bars for cities (show legend)
    for exp in experience_levels:
        fig.add_trace(go.Bar(
            x=city_exp.index,
            y=city_exp[exp],
            name=exp,
            legendgroup=exp,
            marker_color=color_map[exp],
            showlegend=True,
            marker_line_width=0
        ), row=1, col=1)

    # Add stacked bars for states (hide legend)
    for exp in experience_levels:
        fig.add_trace(go.Bar(
            x=state_exp.index,
            y=state_exp[exp],
            name=exp,
            legendgroup=exp,
            marker_color=color_map[exp],
            showlegend=False,
            marker_line_width=0
        ), row=1, col=2)

    # Update layout and axes
    fig.update_layout(
        barmode='stack',
        height=600,
        width=1100,
        showlegend=True,
        #title_text='<b>Work Experience Level Distribution</b>',
        title_x=0.5
    )
    for i in range(1, 3):
        fig.update_xaxes(tickangle=45, tickfont=dict(size=12), row=1, col=i)
        fig.update_yaxes(title_text='Job Postings', row=1, col=i)

    return pio.to_html(fig, full_html=False)

def generate_top_skills_cities_grouped_bar(df):
    """
    Generates an interactive grouped bar chart showing top 10 skills required
    for job postings across top 10 cities.
    Returns an HTML string for embedding in Flask templates.
    """
    # Group data by city and skill_name
    location_skill = df.groupby(['city', 'skill_name']).size().unstack(fill_value=0)
    city_total = location_skill.sum(axis=1)
    top_cities = city_total.sort_values(ascending=False).head(10).index
    location_skill_top = location_skill.loc[top_cities]
    skill_total = location_skill_top.sum(axis=0)
    top_skills = skill_total.sort_values(ascending=False).head(10).index
    final_data = location_skill_top[top_skills]

    cities = final_data.index.tolist()
    skills = final_data.columns.tolist()

    fig = go.Figure()

    for skill in skills:
        fig.add_trace(go.Bar(
            x=cities,
            y=final_data[skill],
            name=skill
        ))

    fig.update_layout(
        #title='<b>Top 10 Skills in Top 10 Cities (Job Count by Skill and City)</b>',
        title_x=0.5,
        xaxis_title='City',
        yaxis_title='No of Jobs',
        barmode='group',
        xaxis_tickangle=-45,
        legend_title='<b>Skill</b>',
        width=max(1000, 120 * len(cities)),
        height=max(600, 40 * len(skills)),
        legend=dict(
            x=0.95,
            y=0.95,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        )
    )
    fig.update_yaxes(type='log')

    return pio.to_html(fig, full_html=False)

def generate_top_skills_distribution_bars(df):
    """
    Generates matplotlib stacked bar charts showing the distribution of top 10 skills
    in the top 10 cities and top 10 states.
    Returns the image as a base64 encoded PNG string for embedding in Flask.
    """
    # Get top 10 cities and states by total job postings
    top_10_cities = df['city'].value_counts().nlargest(10).index
    top_10_states = df['state'].value_counts().nlargest(10).index

    # Filter dataframe for top locations
    df_cities_top = df[df['city'].isin(top_10_cities)]
    df_states_top = df[df['state'].isin(top_10_states)]

    # Group by city/state and skill_name
    city_skill_top = df_cities_top.groupby(['city', 'skill_name']).size().unstack(fill_value=0)
    state_skill_top = df_states_top.groupby(['state', 'skill_name']).size().unstack(fill_value=0)

    # Get top 10 skills by total counts
    top_skills_cities = city_skill_top.sum(axis=0).nlargest(10).index
    top_skills_states = state_skill_top.sum(axis=0).nlargest(10).index

    # Filter to top 10 skills
    city_skill_top_10 = city_skill_top[top_skills_cities]
    state_skill_top_10 = state_skill_top[top_skills_states]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    city_skill_top_10.plot(kind='bar', stacked=True, colormap='Set2', ax=axs[0])
    #axs[0].set_title('Top 10 Skills Distribution - Top 10 Cities', fontsize=18, fontweight='bold')
    axs[0].set_xlabel('City')
    axs[0].set_ylabel('Job Postings')
    axs[0].tick_params(axis='x', rotation=45)

    state_skill_top_10.plot(kind='bar', stacked=True, colormap='Set2', ax=axs[1])
    #axs[1].set_title('Top 10 Skills Distribution - Top 10 States', fontsize=18, fontweight='bold')
    axs[1].set_xlabel('State')
    axs[1].set_ylabel('Job Postings')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Convert plot to PNG image encoded in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64

def generate_country_worktype_distribution(df):
    """
    Generates an interactive grouped bar chart showing job type distribution by country.
    Uses a sequential color palette and sets y-axis to log scale.
    Returns an HTML div string for embedding in Flask templates.
    """
    work_types = df['formatted_work_type'].unique()
    countries = df['country'].unique()
    
    palette = px.colors.sequential.Agsunset
    colors = [palette[int(i * (len(palette)-1) / (len(work_types)-1))] for i in range(len(work_types))]
    
    fig = go.Figure()

    for i, work_type in enumerate(work_types):
        values = [df[(df['country'] == country) & (df['formatted_work_type'] == work_type)].shape[0] for country in countries]
        fig.add_trace(go.Bar(
            name=work_type,
            x=countries,
            y=values,
            marker_color=colors[i % len(colors)],
        ))
    
    fig.update_layout(
        barmode='group',
        #title='<b>Country wise Work Types (Log Scale)</b>',
        title_x=0.5,
        xaxis_title='Country',
        yaxis_title='Number of Jobs',
        legend_title='Work Type',
    )
    
    fig.update_yaxes(type='log')
    
    return pio.to_html(fig, full_html=False)

def generate_industry_distribution_piecharts(df):
    """
    Generates matplotlib pie charts showing share of top industries in top cities and states.
    Returns the image as a base64 encoded PNG string for embedding in Flask.
    """
    # Get top 10 cities and states by total job postings
    top_10_cities_series = df['city'].value_counts().nlargest(10)
    top_10_cities = top_10_cities_series.index
    top_10_states_series = df['state'].value_counts().nlargest(10)
    top_10_states = top_10_states_series.index

    # Filter dataframe for top locations
    df_cities_top = df[df['city'].isin(top_10_cities)]
    df_states_top = df[df['state'].isin(top_10_states)]

    # Group by industry name for cities and states
    city_industry_top = df_cities_top.groupby(['city', 'industry_name']).size().unstack(fill_value=0)
    state_industry_top = df_states_top.groupby(['state', 'industry_name']).size().unstack(fill_value=0)

    # Get top 10 industries by total counts
    top_industries_cities = city_industry_top.sum(axis=0).nlargest(10).index
    top_industries_states = state_industry_top.sum(axis=0).nlargest(10).index

    # Filter to top 10 industries
    city_industry_top_10 = city_industry_top[top_industries_cities]
    state_industry_top_10 = state_industry_top[top_industries_states]

    # Create 'Other' category (sum of industries outside top 10)
    city_industry_other = city_industry_top.drop(columns=top_industries_cities, errors='ignore').sum(axis=1)
    state_industry_other = state_industry_top.drop(columns=top_industries_states, errors='ignore').sum(axis=1)

    # Prepare pie chart data
    city_industry_totals = city_industry_top_10.sum(axis=0)
    city_industry_other_total = city_industry_other.sum()
    city_pie_data_final = pd.concat([city_industry_totals, pd.Series({'Other': city_industry_other_total})])

    state_industry_totals = state_industry_top_10.sum(axis=0)
    state_industry_other_total = state_industry_other.sum()
    state_pie_data_final = pd.concat([state_industry_totals, pd.Series({'Other': state_industry_other_total})])

    # Plot pie charts
    fig_pies, axs_pies = plt.subplots(1, 2, figsize=(20, 10))

    if not city_pie_data_final.empty and city_pie_data_final.sum() > 0:
        city_pie_data_final.plot.pie(
            ax=axs_pies[0],
            fontsize=14,
            autopct='%1.1f%%',
            startangle=90,
            colormap='tab20',
            pctdistance=0.85
        )
        axs_pies[0].set_title('For Top Cities', fontsize=18, fontweight='bold')
        axs_pies[0].set_ylabel('')

    if not state_pie_data_final.empty and state_pie_data_final.sum() > 0:
        state_pie_data_final.plot.pie(
            ax=axs_pies[1],
            fontsize=14,
            autopct='%1.1f%%',
            startangle=90,
            colormap='tab20',
            pctdistance=0.85
        )
        axs_pies[1].set_title('For Top States', fontsize=18, fontweight='bold')
        axs_pies[1].set_ylabel('')

    plt.tight_layout()

    # Convert plot to PNG image encoded in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig_pies)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64

def generate_location_piecharts(df):
    """
    Generates matplotlib pie charts visualizing job distribution by location:
    - Top 10 states + Other
    - Top 5 cities + Other
    Returns the image as base64 encoded PNG string suitable for embedding in HTML.
    """
    # Data preparation for states
    top_states = df['state'].value_counts().nlargest(10)
    top_state_names = top_states.index.tolist()
    state_counts = df['state'].value_counts()
    top_state_counts = state_counts.loc[top_state_names]
    other_states_count = state_counts.loc[~state_counts.index.isin(top_state_names)].sum()
    state_pie_data = pd.concat([top_state_counts, pd.Series({'Other': other_states_count})])

    # Data preparation for cities
    top_cities = df['city'].value_counts().nlargest(5)
    top_city_names = top_cities.index.tolist()
    city_counts = df['city'].value_counts()
    top_city_counts = city_counts.loc[top_city_names]
    other_cities_count = city_counts.loc[~city_counts.index.isin(top_city_names)].sum()
    city_pie_data = pd.concat([top_city_counts, pd.Series({'Other': other_cities_count})])

    # Plot pie charts side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Left Pie Chart (States)
    colors_states = plt.cm.Set2.colors[:len(state_pie_data)]
    wedges_s, texts_s, autotexts_s = axs[0].pie(
        state_pie_data.values,
        labels=state_pie_data.index, # Labels (state names) outside
        autopct='%1.1f%%',
        colors=colors_states,
        startangle=90,
        counterclock=True,
        pctdistance=0.85, # Percentage labels inside
        labeldistance=1.1 # Labels (state names) outside
    )
    # Set percentage font size
    for autotext in autotexts_s:
        autotext.set_fontsize(11)
    # Set label font size
    for text in texts_s:
        text.set_fontsize(12)
    axs[0].set_title('By States', fontsize=14, fontweight='bold') # Remove title for states chart
    axs[0].set_ylabel('')

    # Right Pie Chart (Cities)
    colors_cities = plt.cm.Set3.colors[:len(city_pie_data)]
    wedges_c, texts_c, autotexts_c = axs[1].pie(
        city_pie_data.values,
        labels=city_pie_data.index, # Labels (city names) outside
        autopct='%1.1f%%',
        colors=colors_cities,
        startangle=90,
        counterclock=False, # Clockwise direction
        pctdistance=0.85, # Percentage labels inside
        labeldistance=1.1 # Labels (city names) outside
    )
    
    for text in texts_c:
        text.set_fontsize(12)
    axs[1].set_title('By Cities', fontsize=14, fontweight='bold') # Remove title for cities chart
    axs[1].set_ylabel('')

    plt.tight_layout()

    # Save plot to PNG image in memory and encode as base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64



@app.route('/')
def index():
    df = load_data()
    img_jobtitle = generate_wordcloud(' '.join(df['title'].astype(str)), 'Set1')
    img_skills = generate_wordcloud(' '.join(df['skill_name'].astype(str)), 'tab10')
    img_industries = generate_wordcloud(' '.join(df['industry_name'].astype(str)), 'magma')
    plotly_boxplots_html = generate_interactive_boxplot_html(df)
    histograms = generate_histograms(df)
    #barplots = generate_barplots(df)
    top10_categorical_html = generate_top10_categorical_barplots(df)
    cat_dist_html = generate_categorical_distribution_bars(df)
    worktype_explevel_html = generate_worktype_explevel_distribution(df)
    explevel_worktype_html = generate_explevel_worktype_distribution(df)
    top_locations_html = generate_top_locations_barplots(df)
    worktype_dist_html = generate_worktype_distribution_stacked(df)
    workexp_dist_html = generate_workexperience_distribution_stacked(df)
    top_skills_cities_html = generate_top_skills_cities_grouped_bar(df)
    top_skills_dist_img = generate_top_skills_distribution_bars(df)
    country_worktype_html = generate_country_worktype_distribution(df)
    industry_dist_img = generate_industry_distribution_piecharts(df)
    location_pie_img = generate_location_piecharts(df)


        
    return render_template('index.html',
                           img_jobtitle=img_jobtitle,
                           img_skills=img_skills,
                           img_industries=img_industries,
                           histograms=histograms,
                           plotly_boxplots_html=plotly_boxplots_html,
                           #barplots=barplots,
                           top10_categorical_html=top10_categorical_html,
                           cat_dist_html=cat_dist_html,
                           worktype_explevel_html =worktype_explevel_html,
                           explevel_worktype_html=explevel_worktype_html,
                           top_locations_html=top_locations_html,
                           worktype_dist_html=worktype_dist_html,
                           workexp_dist_html=workexp_dist_html,
                           top_skills_cities_html=top_skills_cities_html,
                           top_skills_dist_img=top_skills_dist_img,
                           country_worktype_html=country_worktype_html, 
                           industry_dist_img=industry_dist_img,
                           location_pie_img=location_pie_img,
                           )

if __name__ == '__main__':
    app.run(debug=True)

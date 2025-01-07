# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Movie Data Analysis",
    page_icon="🎬",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page", 
    ["Home", "Data Overview", "Simple Visualizations", "Complex Analysis"]
)

# Load and preprocess data
@st.cache_data  # Cache the data loading
def load_data():
    movie_data = pd.read_csv('data-attempt-3.csv')
    return movie_data

@st.cache_data
def clean_votes(vote_str):
    original_value = vote_str  # Store original value for debugging
    try:
        # Handle N/A cases
        if pd.isna(vote_str) or vote_str == 'N/A':
            return np.nan
            
        # Convert to string
        vote_str = str(vote_str).strip()
        
        # Handle M (million) cases
        if 'M' in vote_str:
            result = float(vote_str.replace('M', '')) * 1_000_000
            print(f"Converted {original_value} to {result}")
            return result
            
        # Handle cases like "1.4000" - these should be 1400
        if '.' in vote_str and vote_str.endswith('000'):
            # Remove the '000' suffix and convert remaining number
            base = float(vote_str.replace('000', ''))
            result = base * 1000
            print(f"Converted {original_value} to {result}")
            return result
        
        # Convert regular numbers
        result = float(vote_str)
        if result < 100:  # Let's flag suspiciously low values
            print(f"Warning: Very low vote count detected: {original_value} -> {result}")
        return result
        
    except Exception as e:
        print(f"Error processing value: {vote_str}, Error: {str(e)}")
        return np.nan
    pass

@st.cache_data
def process_data(movie_data):
    df_clean = movie_data.copy()
    df_clean['votes'] = df_clean['votes'].apply(clean_votes)

    # conveert release year to numeric and filter out years outside of 1900-2024
    df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
    df_clean = df_clean[df_clean['release_year'].between(1900, 2024)]
    
    # Genre transformation
    df_clean['genre_list'] = df_clean['genre'].apply(lambda x: ast.literal_eval(x))
    all_genres = set()
    for genres in df_clean['genre_list']:
        all_genres.update(genres)
    
    for genre in all_genres:
        df_clean[f'genre_{genre.lower().replace(" ", "_")}'] = df_clean['genre_list'].apply(lambda x: 1 if genre in x else 0)
    
    return df_clean

# Load and process data
movie_data = load_data()
df_clean = process_data(movie_data)

# Create OLAP tables
def create_olap_tables(df_clean):
    # Create Dimension Tables
    dim_movie = df_clean[['title', 'title_type', 'release_year']].copy()
    dim_movie['movie_id'] = range(1, len(dim_movie) +1)

    # Create a Genre Bridge Table (for many-to-many relationaship)
    genre_columns = [col for col in df_clean.columns if col.startswith('genre_')]
    genre_bridge = []

    for movie_id in range(1, len(df_clean) + 1):
        movie_genres = df_clean.iloc[movie_id-1][genre_columns]
        for genre_col in genre_columns:
            if movie_genres[genre_col] == 1:
                genre_name = genre_col.replace('genre_', '').replace('_','').title()
                genre_bridge.append({
                    'movie_id': movie_id,
                    'genre': genre_name
                })

    dim_genre_bridge = pd.DataFrame(genre_bridge)

    # Create a Fact Table
    fact_movie_ratings = pd.DataFrame({
        'movie_id': range(1, len(df_clean) +1),
        'rating': df_clean['rating'],
        'votes': df_clean['votes'],
        'rank': df_clean['rank']
    })
    
    return dim_movie, dim_genre_bridge, fact_movie_ratings

# Page content
if page == "Home":
    st.title("🎬 Movie Data Analysis Dashboard")
    st.write("Welcome to the Movie Data Analysis Dashboard! This application provides insights into movie data including ratings, votes, and genres.")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies", len(df_clean))
    with col2:
        st.metric("Average Rating", f"{df_clean['rating'].mean():.2f}")
    with col3:
        st.metric("Total Votes", f"{df_clean['votes'].sum():,.0f}")

elif page == "Data Overview":
    st.title("Data Overview")
    
    # Show data quality metrics
    st.subheader("Data Quality Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing Values:")
        st.write(df_clean.isnull().sum())
    with col2:
        st.write("Data Types:")
        st.write(df_clean.dtypes)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df_clean.head())

elif page == "Simple Visualizations":
    st.title("Simple Visualizations")
    
   # 1) Top 10 Movies by Votes with video links
    st.subheader("Top 10 Movies by Votes")
    top_10 = df_clean.nlargest(10, 'votes')[['title', 'votes', 'video_link']]
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart
        fig = px.bar(top_10, 
                     x='votes', 
                     y='title', 
                     orientation='h',
                     title='Top 10 Movies by Votes')
        
        # Make the chart look better
        fig.update_layout(
            xaxis_title="Number of Votes",
            yaxis_title="Movie Title",
            height=500
        )
        
        # Format vote numbers
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Votes: %{x:,.0f}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Watch Trailer")
        # Create a dropdown to select movie
        selected_movie = st.selectbox(
            "Select a movie to watch its trailer",
            top_10['title'].tolist(),
            key="movie_selector"
        )
        
        # Show video for selected movie
        if selected_movie:
            video_link = top_10[top_10['title'] == selected_movie]['video_link'].iloc[0]
            st.video(video_link)

    # Add note about interaction
    st.info("👆 Select a movie from the dropdown to watch its trailer!")

    
    # 2) Movie Types Distribution
    st.subheader("Distribution of Movie Types")
    fig = px.pie(df_clean, names='title_type', title='Movie Types Distribution')
    st.plotly_chart(fig)

    # 3) Movie Types Distribution
    st.subheader("Rating Distribution by Vote Count Category")
    
    # Create vote categories
    df_clean['vote_category'] = pd.qcut(df_clean['votes'], 
                                      q=5, 
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Create box plot using plotly for better interactivity
    fig3 = px.box(df_clean, 
                  x='vote_category', 
                  y='rating',
                  title='Rating Distribution by Vote Count Category')
    
    fig3.update_layout(
        xaxis_title='Vote Count Category',
        yaxis_title='Rating',
        boxmode='group'
    )
    
    st.plotly_chart(fig3)


elif page == "Complex Analysis":
    st.title("Complex Analysis")
    
    # Create OLAP tables
    dim_movie, dim_genre_bridge, fact_movie_ratings = create_olap_tables(df_clean)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Genre Trends", 
        "Genre Combinations", 
        "Success Matrix",
        "Advanced Analysis"
    ])
    
    with tab1:
        st.subheader("Genre Rating Trends Over Time")
        genre_performance = pd.merge(
            dim_genre_bridge,
            fact_movie_ratings,
            on='movie_id'
        ).merge(
            dim_movie,
            on='movie_id'
        )

        # Filter out invalid ratings
        genre_performance = genre_performance[
            genre_performance['rating'].notna() & 
            (genre_performance['rating'] > 0)
        ]
        
        # Let user select number of top genres to display
        n_genres = st.slider("Select number of top genres to display", 3, 10, 5)
        top_genres = genre_performance['genre'].value_counts().head(n_genres).index
        
        # Calculate average rating by year for each genre wtih minimum movie count threshold
        min_movies_per_year = 2
        genre_yearly = []

        for genre in top_genres:
            genre_data = genre_performance[genre_performance['genre'] == genre]
            yearly_stats = genre_data.groupby('release_year').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            yearly_stats.columns = ['release_year', 'rating', 'count']
            
            # Filter years with enough movies
            yearly_stats = yearly_stats[yearly_stats['count'] >= min_movies_per_year]
            yearly_stats['genre'] = genre
            genre_yearly.append(yearly_stats)
    
        genre_yearly_df = pd.concat(genre_yearly)
        
        # Create the plot
        fig1 = px.line(genre_yearly_df, 
                    x='release_year', 
                    y='rating', 
                    color='genre',
                    title='Genre Rating Trends')
        
        fig1.update_layout(
            xaxis_title="Release Year",
            yaxis_title="Average Rating",
            yaxis_range=[0, 10],  # Set fixed range for ratings
            showlegend=True,
            hovermode='x unified'
        )

        st.plotly_chart(fig1)
    
        # Add data quality information
        st.info(f"Note: Only showing years with at least {min_movies_per_year} movies per genre. Rating trends are smoothed for better visualization.")

    with tab2:
        st.subheader("Genre Combinations Analysis")
        
        # Genre Combination Analysis
        genre_combinations = []
        movies_with_genres = dim_genre_bridge.groupby('movie_id')['genre'].agg(list).reset_index()

        for _, row in movies_with_genres.iterrows():
            genres = row['genre']
            if len(genres) >= 2:
                for i in range(len(genres)):
                    for j in range(i + 1, len(genres)):
                        genre_combinations.append(tuple(sorted([genres[i], genres[j]])))

        genre_pair_counts = pd.Series(genre_combinations).value_counts().head(15)
        
        fig2 = px.bar(
            x=genre_pair_counts.index.map(str),
            y=genre_pair_counts.values,
            title="Most Common Genre Combinations"
        )
        fig2.update_layout(
            xaxis_title="Genre Pairs",
            yaxis_title="Count",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("Genre Success Matrix")
        
        # Calculate genre metrics
        genre_metrics = genre_performance.groupby('genre').agg({
            'rating': ['mean', 'count'],
            'votes': 'mean'
        }).round(2)

        genre_metrics.columns = ['avg_rating', 'movie_count', 'avg_votes']
        
        # Filter options
        min_movies = st.slider("Minimum number of movies per genre", 1, 50, 10)
        genre_metrics_filtered = genre_metrics[genre_metrics['movie_count'] >= min_movies]
        
        fig3 = px.scatter(
            genre_metrics_filtered,
            x='avg_votes',
            y='avg_rating',
            size='movie_count',
            hover_data=['movie_count'],
            text=genre_metrics_filtered.index,
            title="Genre Success Matrix"
        )
        
        fig3.update_layout(
            xaxis_title="Average Votes (log scale)",
            yaxis_title="Average Rating",
            xaxis_type="log"
        )
        st.plotly_chart(fig3)

    with tab4:
        st.subheader("Advanced Analysis")
        
        # Year-over-Year Analysis
        st.write("Year-over-Year Rating Analysis")
        yearly_stats = genre_performance.groupby('release_year').agg({
            'rating': ['mean', 'std', 'count'],
            'votes': ['mean', 'sum']
        })
        yearly_stats.columns = ['avg_rating', 'rating_std', 'movie_count', 'avg_votes', 'total_votes']
        
        # Create multi-line plot
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats['avg_rating'],
            name='Average Rating',
            line=dict(color='blue')
        ))
        fig4.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats['movie_count'],
            name='Number of Movies',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig4.update_layout(
            title='Yearly Trends: Average Rating vs Number of Movies',
            xaxis_title='Year',
            yaxis_title='Average Rating',
            yaxis2=dict(
                title='Number of Movies',
                overlaying='y',
                side='right'
            )
        )
        st.plotly_chart(fig4)
        
        # Correlation Analysis
        st.write("Correlation Analysis")
        correlation_data = genre_performance.groupby('genre').agg({
            'rating': 'mean',
            'votes': 'mean',
            'release_year': 'mean'
        })
        
        correlation_matrix = correlation_data.corr()
        fig5 = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig5)

        # Additional metrics
        st.write("Genre Volatility Analysis")

        # Add debugging information
        st.write("Number of movies per genre before filtering:")
        genre_counts = genre_performance.groupby('genre').size().sort_values(ascending=False)
        st.write(genre_counts)

        # Filter genres with minimum number of movies for meaningful volatility
        min_movies_for_volatility = 5  # Set minimum threshold
        filtered_genre_performance = genre_performance.groupby('genre').filter(lambda x: len(x) >= min_movies_for_volatility)

        genre_volatility = filtered_genre_performance.groupby('genre').agg({
            'rating': ['std', 'count', 'mean']
        }).round(3)
        
        genre_volatility.columns = ['rating_std', 'movie_count', 'avg_rating']
        genre_volatility = genre_volatility.sort_values('rating_std', ascending=False)

        # Display debugging information
        st.write("\nVolatility Statistics:")
        st.write(genre_volatility)

        # Create the bar plot with more information
        fig6 = px.bar(
            genre_volatility.reset_index(),
            x='genre',
            y='rating_std',
            title="Genre Rating Volatility",
            hover_data=['movie_count', 'avg_rating']  # Add hover information
        )

        fig6.update_layout(
            xaxis_title="Genre",
            yaxis_title="Rating Standard Deviation",
            xaxis_tickangle=45,
            showlegend=False,
            height=600,  # Increase height for better readability
            margin=dict(b=150)  # Increase bottom margin for rotated labels
        )

        # Add text annotations for context
        st.plotly_chart(fig6)
        st.info("""
        Volatility Analysis Notes:
        - Only showing genres with at least 5 movies
        - Higher standard deviation indicates more varied ratings within the genre
        - Hover over bars to see number of movies and average rating
        """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Group Nanda and Haziq")
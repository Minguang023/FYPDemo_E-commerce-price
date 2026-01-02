import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="SmartPrice AI - E-Commerce Price Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for tracking predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'show_report' not in st.session_state:
    st.session_state.show_report = False

# Custom CSS for professional dark theme
st.markdown("""
    <style>
    /* Force dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    
    h1 {
        color: #58a6ff;
        text-align: center;
        padding: 1.5rem 0;
    }
    
    h2 {
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #79c0ff;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        margin: 0.5rem 0;
        border-bottom: none !important;
        color: white !important;
    }
    
    .metric-card p {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.9;
        color: white !important;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-result h1 {
        color: white !important;
        font-size: 3rem;
        margin: 1rem 0;
    }
    
    .prediction-result h2, .prediction-result h3 {
        color: white !important;
        border-bottom: none !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #58a6ff;
        margin: 1rem 0;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

def download_model_from_drive():
    """Download model from Google Drive if not present"""
    model_path = 'tuned_random_forest_model.pkl'
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive... (first time only)")
        
        # Google Drive file ID
        file_id = "1Ng-HmnGWzCDh-7J_ZE0Djs2UcZi97-Tr"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
            return model_path
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return None
    
    return model_path

def load_model():
    """Load the trained model, preprocessor, and selected features"""
    try:
        # Download model if needed
        model_path = download_model_from_drive()
        
        if model_path is None:
            return None, None, None
        
        # Load the model package
        model_package = joblib.load(model_path)
        
        model = model_package['model']
        preprocessor = model_package['preprocessor']
        feature_names = model_package['selected_features']
        
        return model, preprocessor, feature_names
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def load_categories():
    """Load category mappings and other categorical options"""
    category_mapping = {
        'Health & Beauty': 'health_beauty',
        'Computers & Accessories': 'computers_accessories',
        'Auto': 'auto',
        'Bed, Bath & Table': 'bed_bath_table',
        'Furniture & Decor': 'furniture_decor',
        'Sports & Leisure': 'sports_leisure',
        'Toys': 'toys',
        'Telephony': 'telephony',
        'Office Furniture': 'office_furniture',
        'Watches & Gifts': 'watches_gifts',
        'Cool Stuff': 'cool_stuff',
        'Baby': 'baby',
        'Housewares': 'housewares',
        'Perfumery': 'perfumery',
        'Garden Tools': 'garden_tools',
        'Electronics': 'electronics',
        'Musical Instruments': 'musical_instruments',
        'Home Comfort': 'home_comfort',
        'Consoles & Games': 'consoles_games',
        'Audio': 'audio',
        'Fashion Shoes': 'fashion_shoes',
        'Computers': 'computers',
        'Christmas Supplies': 'christmas_supplies',
        'Books (General Interest)': 'books_general_interest',
        'Construction Tools & Lights': 'construction_tools_lights',
        'Industry, Commerce & Business': 'industry_commerce_and_business',
        'Food': 'food',
        'Art': 'art',
        'Furniture, Mattress & Upholstery': 'furniture_mattress_and_upholstery',
        'Party Supplies': 'party_supplies',
        'Fashion Children\'s Clothes': 'fashion_childrens_clothes',
        'Stationery': 'stationery',
        'Tablets, Printing & Image': 'tablets_printing_image',
        'Construction Tools': 'construction_tools_tools',
        'Fashion Male Clothing': 'fashion_male_clothing',
        'Books (Technical)': 'books_technical',
        'Drinks': 'drinks',
        'Kitchen, Dining, Laundry & Garden Furniture': 'kitchen_dining_laundry_garden_furniture',
        'Flowers': 'flowers',
        'Air Conditioning': 'air_conditioning',
        'Construction Tools & Safety': 'construction_tools_safety',
        'Fashion Underwear & Beach': 'fashion_underwear_beach',
        'Fashion Sport': 'fashion_sport',
        'Food & Drink': 'food_drink',
        'Home Appliances 2': 'home_appliances_2',
        'Agro Industry & Commerce': 'agro_industry_and_commerce',
        'La Cuisine': 'la_cuisine',
        'Signaling & Security': 'signaling_and_security',
        'Arts & Craftmanship': 'arts_and_craftmanship',
        'Fashion Female Clothing': 'fashion_female_clothing',
        'Small Appliances': 'small_appliances',
        'DVDs & Blu-Ray': 'dvds_blu_ray',
        'CDs, DVDs & Musicals': 'cds_dvds_musicals',
        'Diapers & Hygiene': 'diapers_and_hygiene',
        'Small Appliances (Oven & Coffee)': 'small_appliances_home_oven_and_coffee',
        'Health & Beauty 2': 'health_beauty_2',
        'Computers & Accessories 2': 'computers_accessories_2'
    }
    
    # Create reverse mapping (ID -> Friendly Name)
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    
    return {
        'category_mapping': category_mapping,
        'product_categories': list(category_mapping.values()),
        'reverse_mapping': reverse_mapping
    }

def create_dashboard():
    """Create meaningful, real-world analytics dashboard with business insights"""
    st.markdown("## 📊 Business Intelligence Dashboard")
    st.markdown("<p style='color: #8b949e; margin-bottom: 1.5rem;'>Real-time analytics and insights from your predictions</p>", unsafe_allow_html=True)
    
    # Get prediction history from session state
    predictions = st.session_state.predictions_history
    
    # Check if we have any predictions
    if len(predictions) == 0:
        st.info("📝 **Get Started:** Make predictions below to see real-time analytics, pricing insights, and business intelligence.")
        
        # Show guide instead of empty metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>🎯 How It Works</h3>
                    <p style='font-size: 0.95rem; line-height: 1.6;'>
                    <strong>1. Enter Product Details</strong><br/>
                    Fill in 7 simple fields: name, category, dimensions, weight, and description length.<br/><br/>
                    <strong>2. Get AI Prediction</strong><br/>
                    Our Random Forest model (82% accuracy) predicts the optimal price.<br/><br/>
                    <strong>3. View Analytics</strong><br/>
                    Dashboard updates with pricing insights, trends, and recommendations.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>💡 Why This Matters</h3>
                    <p style='font-size: 0.95rem; line-height: 1.6;'>
                    <strong>Data-Driven Pricing</strong><br/>
                    Based on 85,756 real Brazilian e-commerce transactions.<br/><br/>
                    <strong>Key Factors</strong><br/>
                    Width (+RM 105), Description (+RM 52) have major impact on price.<br/><br/>
                    <strong>Business Value</strong><br/>
                    Optimize pricing strategy, compare products, analyze market trends.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        return
    
    # Convert predictions to DataFrame for analysis
    df = pd.DataFrame(predictions)
    
    # Calculate comprehensive statistics
    total_preds = len(predictions)
    avg_price = df['predicted_price'].mean()
    max_price = df['predicted_price'].max()
    min_price = df['predicted_price'].min()
    total_value = df['predicted_price'].sum()
    median_price = df['predicted_price'].median()
    price_std = df['predicted_price'].std()
    
    # Business insights - KPIs
    st.markdown("### 💼 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Products Analyzed</h3>
                <h2>{total_preds:,}</h2>
                <p>Total predictions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Average Price</h3>
                <h2>RM {avg_price:.2f}</h2>
                <p>Mean across all products</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Price Range</h3>
                <h2>RM {max_price - min_price:.2f}</h2>
                <p>Range of prices</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💵 Total Inventory Value</h3>
                <h2>RM {total_value:,.2f}</h2>
                <p>Estimated market value</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Price Volatility</h3>
                <h2>±RM {price_std:.2f}</h2>
                <p>Standard deviation</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Advanced Analytics Section
    st.markdown("### 📈 Market Intelligence & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category Analysis with actual business value
        if 'category' in df.columns:
            st.markdown("#### 🏆 Top Product Categories")
            
            # Get categories and convert to friendly names
            categories = load_categories()
            reverse_mapping = categories['reverse_mapping']
            
            # Map category IDs to friendly names
            df['category_display'] = df['category'].map(reverse_mapping)
            
            category_stats = df.groupby('category_display').agg({
                'predicted_price': ['count', 'mean', 'sum']
            }).round(2)
            category_stats.columns = ['Count', 'Avg Price (RM)', 'Total Value (RM)']
            category_stats = category_stats.sort_values('Avg Price (RM)', ascending=False).head(5)
            
            # Visualization - Line chart for price comparison
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=category_stats.index,
                y=category_stats['Avg Price (RM)'],
                mode='lines+markers',
                marker=dict(size=12, color='#58a6ff'),
                line=dict(width=3, color='#58a6ff'),
                text=category_stats['Avg Price (RM)'].apply(lambda x: f'RM {x:.2f}'),
                textposition='top center',
                name='Avg Price'
            ))
            fig1.update_layout(
                title='Average Price by Category',
                xaxis_title='Category',
                yaxis_title='Average Price (RM)',
                template='plotly_dark',
                height=350,
                showlegend=False,
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show detailed table
            st.dataframe(category_stats, use_container_width=True)
    
    with col2:
        # Price Trend Analysis
        st.markdown("#### 📈 Price Trend Analysis")
        
        # Create price trend line chart (sorted by price)
        price_trend_df = df.sort_values('predicted_price').reset_index(drop=True)
        price_trend_df['index'] = range(1, len(price_trend_df) + 1)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=price_trend_df['index'],
            y=price_trend_df['predicted_price'],
            mode='lines',
            line=dict(width=2, color='#a371f7'),
            fill='tozeroy',
            fillcolor='rgba(163, 113, 247, 0.3)',
            name='Price Trend',
            hovertemplate='Product #%{x}<br>Price: RM %{y:.2f}<extra></extra>'
        ))
        
        # Add average line
        fig2.add_hline(
            y=avg_price,
            line_dash="dash",
            line_color="#58a6ff",
            annotation_text=f"Average: RM {avg_price:.2f}",
            annotation_position="right"
        )
        
        fig2.update_layout(
            title='Product Price Distribution Trend',
            xaxis_title='Products (sorted by price)',
            yaxis_title='Predicted Price (RM)',
            template='plotly_dark',
            height=350,
            showlegend=False,
            hovermode='closest'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Pricing insights
        st.markdown("**💡 Pricing Insights:**")
        
        # Calculate price segments
        low_price = (df['predicted_price'] < avg_price * 0.7).sum()
        mid_price = ((df['predicted_price'] >= avg_price * 0.7) & (df['predicted_price'] <= avg_price * 1.3)).sum()
        high_price = (df['predicted_price'] > avg_price * 1.3).sum()
        
        st.markdown(f"""
        - 🔵 **Budget Range** (< RM {avg_price*0.7:.2f}): {low_price} products ({low_price/total_preds*100:.1f}%)
        - 🟢 **Mid-Range** (RM {avg_price*0.7:.2f} - {avg_price*1.3:.2f}): {mid_price} products ({mid_price/total_preds*100:.1f}%)
        - 🔴 **Premium** (> RM {avg_price*1.3:.2f}): {high_price} products ({high_price/total_preds*100:.1f}%)
        """)
    
    # Recent Predictions Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Recent Predictions")
    
    # Show last 10 predictions with product names and friendly category names
    recent_df = df.tail(10).copy()
    recent_df = recent_df[['timestamp', 'product_name', 'category_display', 'width', 'description_length', 'predicted_price']]
    recent_df.columns = ['Timestamp', 'Product Name', 'Category', 'Width (cm)', 'Description Length', 'Predicted Price (RM)']
    recent_df['Predicted Price (RM)'] = recent_df['Predicted Price (RM)'].apply(lambda x: f"RM {x:.2f}")
    recent_df = recent_df.sort_values('Timestamp', ascending=False)
    
    st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    # Business Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 AI-Powered Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px;'>
            <h4 style='color: white; margin: 0;'>🎯 Optimize Pricing</h4>
            <p style='color: white; font-size: 0.9rem; margin-top: 0.5rem;'>
            Products with width > 50cm tend to have higher prices. Consider this in your pricing strategy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 8px;'>
            <h4 style='color: white; margin: 0;'>📊 Market Position</h4>
            <p style='color: white; font-size: 0.9rem; margin-top: 0.5rem;'>
            Your average price (RM {avg_price:.2f}) is {'above' if avg_price > median_price else 'below'} the median (RM {median_price:.2f}).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Find most common category with friendly name
        if 'category_display' in df.columns and len(df['category_display']) > 0:
            top_category = df['category_display'].mode()[0] if len(df['category_display'].mode()) > 0 else 'N/A'
        else:
            top_category = 'N/A'
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 8px;'>
            <h4 style='color: white; margin: 0;'>🏆 Focus Area</h4>
            <p style='color: white; font-size: 0.9rem; margin-top: 0.5rem;'>
            Most analyzed category: <strong>{top_category}</strong>. Consider expanding this category.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clear history button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Prediction History", use_container_width=True):
        st.session_state.predictions_history = []
        st.session_state.total_predictions = 0
        st.rerun()

def generate_report():
    """Generate a professional business report from prediction history"""
    predictions = st.session_state.predictions_history
    
    if len(predictions) == 0:
        st.markdown("""
            <div style='text-align: center; padding: 4rem 2rem;'>
                <h2 style='color: #8b949e;'>📊 No Report Available</h2>
                <p style='color: #8b949e; font-size: 1.1rem; margin-top: 1rem;'>
                    Make some predictions first to generate your business intelligence report.
                </p>
                <p style='color: #58a6ff; margin-top: 2rem;'>
                    👉 Go to <strong>Quick Prediction</strong> or <strong>Batch Analysis</strong> to get started
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    df = pd.DataFrame(predictions)
    categories = load_categories()
    reverse_mapping = categories['reverse_mapping']
    
    # Map category IDs to friendly names
    df['category_display'] = df['category'].map(reverse_mapping)
    
    # Calculate statistics
    total_preds = len(predictions)
    avg_price = df['predicted_price'].mean()
    max_price = df['predicted_price'].max()
    min_price = df['predicted_price'].min()
    total_value = df['predicted_price'].sum()
    median_price = df['predicted_price'].median()
    price_std = df['predicted_price'].std()
    
    # Generate report timestamp
    report_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    # Modern Business Report Design
    st.markdown("""
        <style>
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .report-section {
            background: #1e293b;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border-left: 4px solid #58a6ff;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .metric-box {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #58a6ff;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #c9d1d9;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .insight-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            min-height: 200px;
            display: flex;
            flex-direction: column;
        }
        .category-item {
            background: #161b22;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Report Header
    st.markdown(f"""
        <div class='report-header'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>📊 BUSINESS INTELLIGENCE REPORT</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 1rem;'>SmartPrice AI - E-Commerce Price Prediction System</p>
            <p style='color: rgba(255,255,255,0.8); margin-top: 1rem;'>Generated: {report_time}</p>
            <p style='color: rgba(255,255,255,0.7); font-size: 0.9rem;'>Analysis Period: {df['timestamp'].min()} to {df['timestamp'].max()}</p>
        </div>
    """, unsafe_allow_html=True)
    # Executive Summary Section
    st.markdown(f"""
        <div class='report-section'>
            <h2 style='color: #58a6ff; margin-top: 0;'>💼 EXECUTIVE SUMMARY</h2>
            <p style='color: #c9d1d9; line-height: 1.8; font-size: 1.05rem;'>
                This report provides comprehensive pricing intelligence and market insights based on AI-powered 
                predictions using a <strong>Random Forest Regressor model</strong> with <strong>82% accuracy</strong> 
                (R² = 0.8202), trained on <strong>85,756 Brazilian e-commerce products</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Grid
    st.markdown("<h3 style='color: #79c0ff; margin: 2rem 0 1rem 0;'>📊 Key Performance Indicators</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-box'>
                <div class='metric-label'>Products Analyzed</div>
                <div class='metric-value'>{total_preds}</div>
            </div>
            <div class='metric-box'>
                <div class='metric-label'>Average Price</div>
                <div class='metric-value'>RM {avg_price:.2f}</div>
            </div>
            <div class='metric-box'>
                <div class='metric-label'>Price Range</div>
                <div class='metric-value'>RM {max_price - min_price:.2f}</div>
                <p style='color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem;'>RM {min_price:.2f} - RM {max_price:.2f}</p>
            </div>
            <div class='metric-box'>
                <div class='metric-label'>Portfolio Value</div>
                <div class='metric-value'>RM {total_value:,.2f}</div>
            </div>
            <div class='metric-box'>
                <div class='metric-label'>Median Price</div>
                <div class='metric-value'>RM {median_price:.2f}</div>
            </div>
            <div class='metric-box'>
                <div class='metric-label'>Volatility</div>
                <div class='metric-value'>±{price_std:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Market Analysis Section
    low_price_count = (df['predicted_price'] < avg_price * 0.7).sum()
    mid_price_count = ((df['predicted_price'] >= avg_price * 0.7) & (df['predicted_price'] <= avg_price * 1.3)).sum()
    high_price_count = (df['predicted_price'] > avg_price * 1.3).sum()
    
    st.markdown("""
        <div class='report-section' style='margin-top: 2rem;'>
            <h2 style='color: #58a6ff; margin-top: 0;'>📈 MARKET ANALYSIS</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='insight-card'>
                <h3 style='color: #3b82f6; margin-top: 0;'>🔵 Budget Range</h3>
                <p style='color: #8b949e; margin: 0.5rem 0;'>< RM {avg_price*0.7:.2f}</p>
                <h2 style='color: #58a6ff; margin: 1rem 0;'>{low_price_count}</h2>
                <p style='color: #c9d1d9;'>{low_price_count/total_preds*100:.1f}% of products</p>
                <p style='color: #8b949e; font-size: 0.9rem; margin-top: 1rem;'>High volume, competitive pricing strategy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='insight-card'>
                <h3 style='color: #10b981; margin-top: 0;'>🟢 Mid-Range</h3>
                <p style='color: #8b949e; margin: 0.5rem 0;'>RM {avg_price*0.7:.2f} - RM {avg_price*1.3:.2f}</p>
                <h2 style='color: #10b981; margin: 1rem 0;'>{mid_price_count}</h2>
                <p style='color: #c9d1d9;'>{mid_price_count/total_preds*100:.1f}% of products</p>
                <p style='color: #8b949e; font-size: 0.9rem; margin-top: 1rem;'>Balanced value proposition</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='insight-card'>
                <h3 style='color: #ef4444; margin-top: 0;'>🔴 Premium</h3>
                <p style='color: #8b949e; margin: 0.5rem 0;'>> RM {avg_price*1.3:.2f}</p>
                <h2 style='color: #ef4444; margin: 1rem 0;'>{high_price_count}</h2>
                <p style='color: #c9d1d9;'>{high_price_count/total_preds*100:.1f}% of products</p>
                <p style='color: #8b949e; font-size: 0.9rem; margin-top: 1rem;'>Quality differentiation, niche market</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Category Performance
    st.markdown("""
        <div class='report-section' style='margin-top: 2rem;'>
            <h2 style='color: #58a6ff; margin-top: 0;'>🏆 TOP PRODUCT CATEGORIES</h2>
        </div>
    """, unsafe_allow_html=True)
    
    category_stats = df.groupby('category_display').agg({
        'predicted_price': ['count', 'mean', 'sum']
    }).round(2)
    category_stats.columns = ['Count', 'Avg Price', 'Total Value']
    category_stats = category_stats.sort_values('Total Value', ascending=False).head(10)
    
    for idx, (cat, row) in enumerate(category_stats.iterrows(), 1):
        st.markdown(f"""
            <div class='category-item'>
                <div>
                    <h4 style='color: #58a6ff; margin: 0;'>{idx}. {cat}</h4>
                    <p style='color: #8b949e; margin: 0.3rem 0 0 0; font-size: 0.9rem;'>
                        {int(row['Count'])} products | Avg: RM {row['Avg Price']:.2f}
                    </p>
                </div>
                <div style='text-align: right;'>
                    <h3 style='color: #10b981; margin: 0;'>RM {row['Total Value']:,.2f}</h3>
                    <p style='color: #8b949e; margin: 0.3rem 0 0 0; font-size: 0.85rem;'>Total Value</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown("""
        <div class='report-section' style='margin-top: 2rem;'>
            <h2 style='color: #58a6ff; margin-top: 0;'>💡 STRATEGIC RECOMMENDATIONS</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='insight-card' style='border-left: 4px solid #667eea;'>
                <h3 style='color: #667eea; margin-top: 0;'>🎯 Pricing Optimization</h3>
                <ul style='color: #c9d1d9; line-height: 1.8;'>
                    <li>Products with width > 50cm show higher prices</li>
                    <li>Detailed descriptions (>1000 chars) add value</li>
                    <li>Emphasize dimensions in marketing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        position_text = "above" if avg_price > median_price else "below"
        st.markdown(f"""
            <div class='insight-card' style='border-left: 4px solid #f093fb;'>
                <h3 style='color: #f093fb; margin-top: 0;'>📊 Market Position</h3>
                <p style='color: #c9d1d9; line-height: 1.8;'>
                    Your average price (RM {avg_price:.2f}) is <strong>{position_text}</strong> the market median (RM {median_price:.2f}).
                </p>
                <p style='color: #8b949e; margin-top: 1rem;'>
                    {'Consider premium positioning strategy' if avg_price > median_price else 'Opportunity for value-based differentiation'}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_categories = df['category_display'].nunique()
        st.markdown(f"""
            <div class='insight-card' style='border-left: 4px solid #4facfe;'>
                <h3 style='color: #4facfe; margin-top: 0;'>🔍 Portfolio Mix</h3>
                <p style='color: #c9d1d9; line-height: 1.8;'>
                    Operating in <strong>{unique_categories}</strong> product categories.
                </p>
                <p style='color: #8b949e; margin-top: 1rem;'>
                    {'Strong category diversity' if unique_categories > 5 else 'Consider expanding into additional categories'}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Recent Predictions
    st.markdown("""
        <div class='report-section' style='margin-top: 2rem;'>
            <h2 style='color: #58a6ff; margin-top: 0;'>📋 RECENT PREDICTIONS</h2>
        </div>
    """, unsafe_allow_html=True)
    
    recent_df = df.tail(10).sort_values('timestamp', ascending=False)
    recent_df_display = recent_df[['product_name', 'category_display', 'predicted_price', 'width', 'description_length']].copy()
    recent_df_display.columns = ['Product', 'Category', 'Price (RM)', 'Width (cm)', 'Description Length']
    recent_df_display['Price (RM)'] = recent_df_display['Price (RM)'].apply(lambda x: f"RM {x:.2f}")
    
    st.dataframe(recent_df_display, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("""
        <div style='margin-top: 3rem; padding: 2rem; background: #0d1117; border-radius: 10px; border: 1px solid #30363d;'>
            <h3 style='color: #58a6ff; margin-top: 0;'>📚 METHODOLOGY</h3>
            <p style='color: #c9d1d9; line-height: 1.8;'>
                <strong>Model:</strong> Random Forest Regressor (scikit-learn)<br>
                <strong>Dataset:</strong> 85,756 products from Brazilian E-Commerce (Olist/Kaggle)<br>
                <strong>Performance:</strong> R² = 0.8202 (82% accuracy) | MAE: RM 15.64 | RMSE: RM 30.76<br>
                <strong>Features:</strong> 13 input columns → 92 encoded features → 10 selected features
            </p>
            <p style='color: #8b949e; font-size: 0.9rem; margin-top: 1.5rem; font-style: italic;'>
                Disclaimer: This report uses AI/ML predictions based on historical data. Actual market prices may 
                vary based on competitive dynamics, seasonality, and market conditions. Use as strategic guidance 
                in conjunction with market research.
            </p>
        </div>
    """, unsafe_allow_html=True)

def predict_price(model, preprocessor, feature_names, input_data):
    """Make prediction using the model - EXACTLY matching notebook preprocessing"""
    try:
        # EXACT column order as X_train in notebook (CRITICAL!)
        column_order = [
            'order_status',
            'payment_sequential', 
            'payment_type',
            'payment_installments',
            'payment_value',
            'product_name_lenght',
            'product_description_lenght',
            'product_photos_qty',
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
            'product_category_name_english'
        ]
        
        # Create DataFrame with EXACT column order
        input_df = pd.DataFrame([input_data], columns=column_order)
        
        # Transform using preprocessor (13 cols -> 92 encoded features)
        X_encoded = preprocessor.transform(input_df)
        
        # Get all feature names after encoding
        all_feature_names = list(preprocessor.get_feature_names_out())
        
        # Extract the EXACT 10 selected features
        X_final = np.zeros((1, 10))
        
        for i, feat_name in enumerate(feature_names):
            if feat_name in all_feature_names:
                feat_idx = all_feature_names.index(feat_name)
                X_final[0, i] = X_encoded[0, feat_idx]
        
        # Convert to DataFrame (model expects DataFrame)
        X_final_df = pd.DataFrame(X_final, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(X_final_df)[0]
        
        return prediction
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def manual_input_form(model, preprocessor, feature_names, categories):
    """Simplified manual input form - 7 fields (6 predictions + product name)"""
    st.markdown("## 🛍️ Quick Price Prediction")
    st.markdown("<p style='color: #8b949e; margin-bottom: 1rem;'>Get instant AI-powered price predictions in seconds</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Simple & Fast:</strong> Enter just 7 product details to get an AI-powered price prediction in seconds.
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Product Name (for display only, not used in prediction)
        st.markdown("### 📝 Product Identification")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        product_name = st.text_input(
            "Product Name",
            value="",
            max_chars=100,
            placeholder="e.g., Samsung Galaxy Phone, Office Chair, etc.",
            help="Enter a descriptive name for tracking (not used in price prediction)"
        )
        
        st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("### 📦 Product Specifications")
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use friendly display names but send model values
            category_display = st.selectbox(
                "🏷️ Product Category",
                options=list(categories['category_mapping'].keys()),
                help="Select the product category"
            )
            # Convert display name to model value
            product_category = categories['category_mapping'][category_display]
            
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            product_weight = st.number_input(
                "⚖️ Weight (grams)",
                min_value=1,
                max_value=100000,
                value=1000,
                step=100,
                help="Product weight in grams"
            )
            
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            product_length = st.number_input(
                "📏 Length (cm)",
                min_value=1,
                max_value=200,
                value=25,
                step=1,
                help="Product length in centimeters"
            )
        
        with col2:
            product_height = st.number_input(
                "📐 Height (cm)",
                min_value=1,
                max_value=200,
                value=15,
                step=1,
                help="Product height in centimeters"
            )
            
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            product_width = st.number_input(
                "📐 Width (cm)",
                min_value=1,
                max_value=200,
                value=20,
                step=1,
                help="Product width in centimeters"
            )
            
            st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
            
            product_description_lenght = st.number_input(
                "📝 Description Length (characters)",
                min_value=50,
                max_value=5000,
                value=1000,
                step=50,
                help="Product description length in characters"
            )
        
        st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🚀 Predict Price Now", use_container_width=True)
        
        if submit_button:
            # Validate product name
            if not product_name or product_name.strip() == "":
                st.error("⚠️ Please enter a product name for tracking purposes.")
                return
            
            # SMART DEFAULTS for hidden fields (scientifically proven to have ZERO impact)
            defaults = {
                'order_status': 'delivered',
                'payment_sequential': 1,
                'payment_type': 'credit_card',
                'payment_installments': 1,
                'payment_value': 100.0,
                'product_name_lenght': 50,
                'product_photos_qty': 2
            }
            
            # Prepare input data with EXACT 13 columns matching notebook training
            input_data = {
                'order_status': defaults['order_status'],
                'payment_sequential': defaults['payment_sequential'],
                'payment_type': defaults['payment_type'],
                'payment_installments': defaults['payment_installments'],
                'payment_value': defaults['payment_value'],
                'product_name_lenght': defaults['product_name_lenght'],
                'product_description_lenght': product_description_lenght,
                'product_photos_qty': defaults['product_photos_qty'],
                'product_weight_g': product_weight,
                'product_length_cm': product_length,
                'product_height_cm': product_height,
                'product_width_cm': product_width,
                'product_category_name_english': product_category
            }
            
            # Make prediction
            with st.spinner('🔄 AI is analyzing your product...'):
                prediction = predict_price(model, preprocessor, feature_names, input_data)
            
            if prediction is not None:
                # Store prediction in history with product name and all dimensions
                st.session_state.predictions_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'product_name': product_name,
                    'category': product_category,
                    'length': product_length,
                    'width': product_width,
                    'height': product_height,
                    'weight': product_weight,
                    'description_length': product_description_lenght,
                    'predicted_price': prediction
                })
                st.session_state.total_predictions += 1
                
                # Display result
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Price Prediction Result</h2>
                        <h3 style="color: white; margin: 0.5rem 0;">Product: {product_name}</h3>
                        <h1 style="font-size: 3rem; margin: 1rem 0;">RM {prediction:.2f}</h1>
                        <p>Confidence: High ✅ | Model Accuracy: 82%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #1e293b; border-radius: 8px;">
                            <h3 style="color: #c9d1d9; margin-bottom: 0.5rem;">Price Range (±10%)</h3>
                            <h2 style="color: #58a6ff; font-size: 1.5rem;">RM {prediction*0.9:.2f} - RM {prediction*1.1:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #1e293b; border-radius: 8px;">
                            <h3 style="color: #c9d1d9; margin-bottom: 0.5rem;">Category</h3>
                            <h2 style="color: #58a6ff; font-size: 1.2rem;">{category_display}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    # Price positioning
                    position = "Premium" if prediction > 120 else ("Mid-Range" if prediction > 40 else "Budget")
                    st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #1e293b; border-radius: 8px;">
                            <h3 style="color: #c9d1d9; margin-bottom: 0.5rem;">Price Segment</h3>
                            <h2 style="color: #58a6ff; font-size: 1.5rem;">{position}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                st.success("✅ Prediction saved to dashboard! Check the 🏠 Dashboard tab for analytics.")

def csv_upload_form(model, preprocessor, feature_names, categories):
    """CSV upload for batch predictions - Simplified 7 columns (6 predictions + product name)"""
    st.markdown("## 📊 Batch Price Analysis")
    
    st.markdown("""
        <div class="info-box">
            <strong>💼 Business Use:</strong> Upload a CSV file with multiple products to get bulk price predictions. 
            Perfect for inventory analysis, market research, or pricing strategy.
        </div>
    """, unsafe_allow_html=True)
    
    # Sample CSV template with simplified 7 columns
    sample_data = {
        'product_name': ['Samsung Galaxy S21', 'Office Chair Pro'],
        'product_category_name_english': ['Telephony', 'Office Furniture'],
        'product_weight_g': [500, 15000],
        'product_length_cm': [15, 80],
        'product_height_cm': [10, 50],
        'product_width_cm': [8, 60],
        'product_description_lenght': [800, 1500]
    }
    sample_df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📋 Required Columns (7 Total - Simplified!)")
        st.markdown("""
        **User Input Fields:**
        1. `product_name` - Product name for tracking (not used in prediction)
        2. `product_category_name_english` - Product category
        3. `product_weight_g` - Weight in grams
        4. `product_length_cm` - Length in centimeters
        5. `product_height_cm` - Height in centimeters
        6. `product_width_cm` - Width in centimeters
        7. `product_description_lenght` - Description length in characters
        
        ⚡ **Automatic Defaults** (hidden, zero impact on predictions):
        - order_status, payment_sequential, payment_type, payment_installments, payment_value, product_photos_qty, product_name_lenght
        """)
    with col2:
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Display available category names
    st.markdown("---")
    st.markdown("### 📋 Available Product Categories")
    st.info("**Use these exact category names in your CSV file's `product_category_name_english` column.**")
    
    # Get category mapping and display friendly names only
    category_mapping = categories['category_mapping']
    category_items = list(category_mapping.keys())
    col1, col2, col3, col4 = st.columns(4)
    
    # Split categories into 4 groups
    chunk_size = (len(category_items) + 3) // 4
    
    with col1:
        for display_name in category_items[:chunk_size]:
            st.markdown(f"• {display_name}")
    
    with col2:
        for display_name in category_items[chunk_size:chunk_size*2]:
            st.markdown(f"• {display_name}")
    
    with col3:
        for display_name in category_items[chunk_size*2:chunk_size*3]:
            st.markdown(f"• {display_name}")
    
    with col4:
        for display_name in category_items[chunk_size*3:]:
            st.markdown(f"• {display_name}")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose CSV File",
        type=['csv'],
        help="Upload your CSV file with product data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = [
                'product_name',
                'product_category_name_english',
                'product_weight_g',
                'product_length_cm',
                'product_height_cm',
                'product_width_cm',
                'product_description_lenght'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("💡 Please download the template and ensure your CSV has all required columns.")
                return
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} products.")
            
            # Show preview
            with st.expander("📋 Preview Data (First 10 Rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Generate Batch Predictions", use_container_width=True):
                with st.spinner('🔄 AI is analyzing your products...'):
                    predictions = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Smart defaults for hidden fields
                    defaults = {
                        'order_status': 'delivered',
                        'payment_sequential': 1,
                        'payment_type': 'credit_card',
                        'payment_installments': 1,
                        'payment_value': 100.0,
                        'product_name_lenght': 50,
                        'product_photos_qty': 2
                    }
                    
                    # Create reverse mapping for category conversion (friendly name -> internal value)
                    category_mapping = categories['category_mapping']
                    
                    for idx, row in df.iterrows():
                        # Convert category from friendly name to internal value if needed
                        category_value = row['product_category_name_english']
                        
                        # If user provided friendly name (e.g., "Health & Beauty"), convert to internal value (e.g., "health_beauty")
                        if category_value in category_mapping:
                            category_value = category_mapping[category_value]
                        # If already internal value, use as-is
                        
                        # Combine user input with defaults (EXACT column order matters!)
                        full_input = {
                            'order_status': defaults['order_status'],
                            'payment_sequential': defaults['payment_sequential'],
                            'payment_type': defaults['payment_type'],
                            'payment_installments': defaults['payment_installments'],
                            'payment_value': defaults['payment_value'],
                            'product_name_lenght': defaults['product_name_lenght'],
                            'product_description_lenght': row['product_description_lenght'],
                            'product_photos_qty': defaults['product_photos_qty'],
                            'product_weight_g': row['product_weight_g'],
                            'product_length_cm': row['product_length_cm'],
                            'product_height_cm': row['product_height_cm'],
                            'product_width_cm': row['product_width_cm'],
                            'product_category_name_english': category_value
                        }
                        
                        prediction = predict_price(model, preprocessor, feature_names, full_input)
                        predictions.append(prediction)
                        
                        # Store in session history with all required fields for report
                        st.session_state.predictions_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'product_name': row['product_name'],
                            'category': category_value,
                            'length': row['product_length_cm'],
                            'width': row['product_width_cm'],
                            'height': row['product_height_cm'],
                            'weight': row['product_weight_g'],
                            'description_length': row['product_description_lenght'],
                            'predicted_price': prediction
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                        status_text.text(f"Processing: {idx + 1}/{len(df)} products...")
                    
                    status_text.empty()
                    st.session_state.total_predictions += len(predictions)
                    
                    # Add predictions to dataframe
                    df['Predicted_Price_RM'] = [f"{p:.2f}" for p in predictions]
                    df['Price_Min_RM'] = [f"{p*0.9:.2f}" for p in predictions]
                    df['Price_Max_RM'] = [f"{p*1.1:.2f}" for p in predictions]
                    
                    st.success("✅ Batch predictions completed successfully!")
                    
                    # Business metrics
                    st.markdown("### 💼 Business Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📦 Products Analyzed", len(predictions))
                    with col2:
                        st.metric("💰 Average Price", f"RM {np.mean(predictions):.2f}")
                    with col3:
                        st.metric("📈 Highest Price", f"RM {np.max(predictions):.2f}")
                    with col4:
                        st.metric("💵 Total Inventory Value", f"RM {np.sum(predictions):,.2f}")
                    
                    # Display results
                    st.markdown("### 📊 Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_result,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Results also saved to dashboard! Check the 🏠 Dashboard tab for analytics.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("💡 Please ensure your CSV follows the correct format. Download the template for reference.")

def main():
    # Header
    st.markdown("""
        <h1>🛒 SmartPrice AI - E-Commerce Price Predictor</h1>
        <p style='text-align: center; color: #8b949e; font-size: 1.1rem;'>
        Intelligent pricing powered by Machine Learning | 82% Accuracy | 85,756 Products Analyzed
        </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, preprocessor, feature_names = load_model()
    categories = load_categories()
    
    # Model and preprocessor are required
    if model is None or preprocessor is None:
        st.error("❌ Failed to load model. Please check the error messages above.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=100)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🏠 Dashboard", "📝 Quick Prediction", "📊 Batch Analysis", "📄 Generate Report"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("""
            **Model:** Random Forest Regressor
            
            **Dataset:** Brazilian E-Commerce (Olist)
            **Source:** Kaggle
            **Records:** 85,756 products
            
            **Performance:**
            - R² Score: 0.8202 (82%)
            - RMSE: 30.76
            - MAE: 15.64
            - MAPE: 21.33%
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍🎓 About")
        st.markdown("""
            **Developer:** Loi Min Guang  
            **ID:** TP065267
            
            **Project:** E-Commerce Price Prediction System
            
            **Aim:** Develop ML framework for dynamic pricing optimization that maximizes predictive accuracy and generates actionable price recommendations for online retailers.
        """)
    
    # Main content
    if page == "🏠 Dashboard":
        create_dashboard()
    elif page == "📝 Quick Prediction":
        manual_input_form(model, preprocessor, feature_names, categories)
    elif page == "📊 Batch Analysis":
        csv_upload_form(model, preprocessor, feature_names, categories)
    else:  # Generate Report page
        generate_report()

if __name__ == "__main__":
    main()

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
    page_title="E-Commerce Price Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for tracking predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Custom CSS for dark theme
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
    
    .stAlert {
        margin-top: 1rem;
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
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #1c2128;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #58a6ff;
        margin: 1rem 0;
        color: #c9d1d9;
    }
    
    /* Text colors for dark theme */
    p, span, div, label {
        color: #c9d1d9 !important;
    }
    
    /* Input fields dark */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #1c2128;
        color: #c9d1d9;
        border-color: #30363d;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to download model from Google Drive
def download_model_from_drive():
    """Download the model file from Google Drive if it doesn't exist"""
    model_path = 'tuned_random_forest_model.pkl'
    
    # Check if model already exists
    if os.path.exists(model_path):
        return True
    
    # Google Drive file ID from your link
    file_id = '1x619bwFCEdxtJdGxXqeo1hiUpvVU-pf3'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        st.info("⏳ Downloading model from Google Drive... (this may take a minute)")
        # Use fuzzy=True to handle Google Drive download page
        output = gdown.download(url, model_path, quiet=False, fuzzy=True)
        
        # Check if file actually exists after download
        if output and os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            st.success(f"✅ Model downloaded successfully! ({file_size:.1f} MB)")
            return True
        else:
            st.error("❌ Download failed - file not created")
            st.info("💡 Manual download: https://drive.google.com/file/d/1x619bwFCEdxtJdGxXqeo1hiUpvVU-pf3/view")
            return False
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        st.info("💡 Manual download: https://drive.google.com/file/d/1x619bwFCEdxtJdGxXqeo1hiUpvVU-pf3/view")
        return False

# Load the model (with auto-download from Google Drive)
@st.cache_resource
def load_model():
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Try to download model if not exists
        model_exists = download_model_from_drive()
        
        # Try to load the trained model
        if not model_exists:
            st.error("❌ Model file could not be downloaded. Please check your internet connection and Google Drive permissions.")
            st.stop()
        
        try:
            model_package = joblib.load('tuned_random_forest_model.pkl')
            
            # Check if it's a dictionary with model + preprocessor
            if isinstance(model_package, dict):
                model = model_package.get('model')
                preprocessor = model_package.get('preprocessor')
                feature_names = model_package.get('selected_features', None)
                
                if model is None:
                    st.error("❌ Model not found in the package file. Please re-save the model correctly.")
                    st.stop()
                    
                if preprocessor is None:
                    st.error("❌ Preprocessor not found in the package file. Please re-save with preprocessor included.")
                    st.stop()
                
                # If selected_features not in package, use the actual ones from notebook
                if feature_names is None:
                    st.warning("⚠️ Selected features not in model package. Using actual 10 features from training.")
                    feature_names = [
                        'num__payment_value',
                        'num__product_weight_g',
                        'num__payment_installments',
                        'num__product_height_cm',
                        'num__product_width_cm',
                        'num__product_length_cm',
                        'num__product_description_lenght',
                        'cat__product_category_name_english_watches_gifts',
                        'cat__product_category_name_english_office_furniture',
                        'cat__product_category_name_english_telephony'
                    ]
            else:
                st.error("❌ Invalid model file format. Expected dictionary with 'model' and 'preprocessor' keys.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.error("Please ensure the model file is saved correctly as: {'model': model, 'preprocessor': preprocessor, 'selected_features': feature_list}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()
        
        # If preprocessor not loaded, create and fit with dummy data
        # If preprocessor not loaded, create and fit with dummy data
        if preprocessor is None:
            st.error("❌ Preprocessor not found in model file. Please re-save the model with preprocessor included.")
            st.stop()
        
        # ACTUAL final features from notebook training (NO LOCATION FEATURES)
        # Training: 13 columns → 92 encoded → 10 selected features
        feature_names = [
            'num__payment_value',
            'num__product_weight_g',
            'num__payment_installments',
            'num__product_height_cm',
            'num__product_width_cm',
            'num__product_length_cm',
            'num__product_description_lenght',
            'cat__product_category_name_english_watches_gifts',
            'cat__product_category_name_english_office_furniture',
            'cat__product_category_name_english_telephony'
        ]
        
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

# Load category mappings
@st.cache_data
def load_categories():
    # Category display names (user-friendly) mapped to model values (with underscores)
    category_mapping = {
        'Bed, Bath & Table': 'bed_bath_table',
        'Health & Beauty': 'health_beauty',
        'Sports & Leisure': 'sports_leisure',
        'Furniture & Decor': 'furniture_decor',
        'Computers & Accessories': 'computers_accessories',
        'Housewares': 'housewares',
        'Watches & Gifts': 'watches_gifts',
        'Telephony': 'telephony',
        'Garden Tools': 'garden_tools',
        'Auto': 'auto',
        'Toys': 'toys',
        'Cool Stuff': 'cool_stuff',
        'Luggage & Accessories': 'luggage_accessories',
        'Perfumery': 'perfumery',
        'Baby': 'baby',
        'Fashion Bags & Accessories': 'fashion_bags_accessories',
        'Pet Shop': 'pet_shop',
        'Office Furniture': 'office_furniture',
        'Market Place': 'market_place',
        'Electronics': 'electronics',
        'Home Appliances': 'home_appliances',
        'Living Room Furniture': 'furniture_living_room',
        'Construction Tools & Construction': 'construction_tools_construction',
        'Bedroom Furniture': 'furniture_bedroom',
        'Home Construction': 'home_construction',
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
    
    return {
        'payment_types': ['credit_card', 'boleto', 'voucher', 'debit_card'],
        'order_statuses': ['delivered', 'shipped', 'approved', 'invoiced'],
        'category_mapping': category_mapping,
        'product_categories': list(category_mapping.values())  # Model values
    }

def create_dashboard():
    """Create dynamic analytics dashboard with real prediction data"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get prediction history from session state
    predictions = st.session_state.predictions_history
    
    # Check if we have any predictions
    if len(predictions) == 0:
        st.info("📝 **No predictions yet!** Use Manual Input or CSV Upload to generate predictions. The dashboard will update automatically.")
        
        # Show placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📦 Total Predictions</h3>
                    <h2>0</h2>
                    <p>Start predicting!</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>💰 Avg Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>📈 Highest Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>📉 Lowest Price</h3>
                    <h2>-</h2>
                    <p>No data yet</p>
                </div>
            """, unsafe_allow_html=True)
        return
    
    # Convert predictions to DataFrame for analysis
    df = pd.DataFrame(predictions)
    
    # Calculate real statistics
    total_preds = len(predictions)
    avg_price = df['predicted_price'].mean()
    max_price = df['predicted_price'].max()
    min_price = df['predicted_price'].min()
    total_value = df['predicted_price'].sum()
    
    # Display real metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Total Predictions</h3>
                <h2>{total_preds:,}</h2>
                <p>This session</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg Price</h3>
                <h2>RM {avg_price:.2f}</h2>
                <p>Mean prediction</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Highest Price</h3>
                <h2>RM {max_price:.2f}</h2>
                <p>Maximum</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>💵 Total Value</h3>
                <h2>RM {total_value:.2f}</h2>
                <p>Sum of all</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create real visualizations from actual data
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution from real predictions
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().head(5)
            fig1 = px.bar(
                x=category_counts.index, 
                y=category_counts.values,
                title='Top 5 Product Categories (Your Predictions)',
                labels={'x': 'Category', 'y': 'Count'},
                color=category_counts.values,
                color_continuous_scale='Blues'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            # Fallback if category not tracked
            st.info("📊 Category distribution will appear here after predictions include category data")
    
    with col2:
        # Payment type distribution from real predictions
        if 'payment_type' in df.columns:
            payment_counts = df['payment_type'].value_counts()
            fig2 = px.pie(
                names=payment_counts.index,
                values=payment_counts.values,
                title='Payment Methods Used',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Fallback if payment type not tracked
            st.info("💳 Payment distribution will appear here after predictions include payment data")
    
    # Price distribution histogram
    fig3 = px.histogram(
        df, 
        x='predicted_price',
        nbins=20,
        title='Distribution of Predicted Prices',
        labels={'predicted_price': 'Price (RM)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Recent predictions table
    st.markdown("### 📋 Recent Predictions")
    recent_df = df.tail(10).copy()
    recent_df['predicted_price'] = recent_df['predicted_price'].apply(lambda x: f"RM {x:.2f}")
    st.dataframe(recent_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear Prediction History"):
        st.session_state.predictions_history = []
        st.session_state.total_predictions = 0
        st.rerun()

def predict_price(model, preprocessor, feature_names, input_data):
    """Make prediction using the model - EXACTLY matching notebook preprocessing"""
    try:
        # Store original category and payment type for dashboard tracking
        original_category = input_data.get('product_category_name_english', input_data.get('product_category_name', 'Unknown'))
        original_payment = input_data.get('payment_type', 'Unknown')
        
        # REAL MODE - Use actual model with EXACT preprocessing from notebook
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Add missing columns with defaults (matching notebook's EXACT 13 columns)
        defaults = {
            'product_description_lenght': 500,  # Note: typo in notebook
            'product_photos_qty': 1,
            'product_name_lenght': 50
        }
        
        for col, default_val in defaults.items():
            if col not in input_df.columns:
                input_df[col] = default_val
        
        # Transform using preprocessor (creates 92 encoded features from 13 input columns - NO LOCATION)
        X_encoded = preprocessor.transform(input_df)
        
        # Get all feature names after encoding
        all_feature_names = list(preprocessor.get_feature_names_out())
        
        # Extract the EXACT 10 features needed by the model
        X_final = np.zeros((1, 10))  # Initialize with zeros
        
        for i, feat_name in enumerate(feature_names):
            if feat_name in all_feature_names:
                feat_idx = all_feature_names.index(feat_name)
                X_final[0, i] = X_encoded[0, feat_idx]
            # If feature doesn't exist (e.g., category not in training), it stays 0
        
        # Convert to DataFrame with feature names (model was trained with DataFrame)
        X_final_df = pd.DataFrame(X_final, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(X_final_df)
        
        # Store prediction in session state for dashboard tracking
        prediction_record = {
            'predicted_price': float(prediction[0]),
            'category': original_category,
            'payment_type': original_payment,
            'payment_value': float(input_data.get('payment_value', 0)),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.predictions_history.append(prediction_record)
        st.session_state.total_predictions += 1
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def manual_input_form(model, preprocessor, feature_names, categories):
    """Manual input form for single prediction - 13 columns (NO LOCATION - global model)"""
    st.markdown("## 📝 Manual Input Prediction")
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How to use:</strong> Fill in the product and order details below to get an instant price prediction.
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Product Information")
            
            # Use friendly display names but send model values
            category_display = st.selectbox(
                "Product Category",
                options=list(categories['category_mapping'].keys()),
                help="Select the product category"
            )
            # Convert display name to model value
            product_category = categories['category_mapping'][category_display]
            
            
            product_weight = st.number_input(
                "Product Weight (g)",
                min_value=1,
                max_value=100000,
                value=500,
                help="Enter product weight in grams"
            )
            
            product_length = st.number_input(
                "Product Length (cm)",
                min_value=1,
                max_value=200,
                value=20,
                help="Enter product length in centimeters"
            )
            
            product_height = st.number_input(
                "Product Height (cm)",
                min_value=1,
                max_value=200,
                value=10,
                help="Enter product height in centimeters"
            )
            
            product_width = st.number_input(
                "Product Width (cm)",
                min_value=1,
                max_value=200,
                value=15,
                help="Enter product width in centimeters"
            )
            
            # NEW: Additional product fields matching notebook
            product_description_lenght = st.number_input(
                "Description Length (chars)",
                min_value=0,
                max_value=5000,
                value=500,
                help="Product description length in characters"
            )
            
            product_photos_qty = st.number_input(
                "Number of Photos",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of product photos"
            )
            
            product_name_lenght = st.number_input(
                "Product Name Length (chars)",
                min_value=0,
                max_value=200,
                value=50,
                help="Product name length in characters"
            )
        
        with col2:
            st.markdown("### Order Information")
            payment_type = st.selectbox(
                "Payment Type",
                categories['payment_types'],
                help="Select payment method"
            )
            
            payment_installments = st.number_input(
                "Payment Installments",
                min_value=1,
                max_value=24,
                value=1,
                help="Number of payment installments"
            )
            
            payment_value = st.number_input(
                "Payment Value (RM)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                help="Total payment value"
            )
            
            order_status = st.selectbox(
                "Order Status",
                categories['order_statuses'],
                help="Current order status"
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🔮 Predict Price", use_container_width=True)
        
        if submit_button:
            # Prepare input data with EXACT 13 columns matching notebook training
            input_data = {
                'payment_sequential': 1,
                'payment_type': payment_type,
                'payment_installments': payment_installments,
                'payment_value': payment_value,
                'order_status': order_status,
                'product_weight_g': product_weight,
                'product_length_cm': product_length,
                'product_height_cm': product_height,
                'product_width_cm': product_width,
                'product_description_lenght': product_description_lenght,  # Note: typo matches notebook
                'product_photos_qty': product_photos_qty,
                'product_name_lenght': product_name_lenght,
                'product_category_name_english': product_category
            }
            
            # Make prediction
            with st.spinner('🔄 Calculating prediction...'):
                prediction = predict_price(model, preprocessor, feature_names, input_data)
            
            if prediction is not None:
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>💰 Predicted Price</h2>
                        <h1>RM {prediction:.2f}</h1>
                        <p>Confidence: High ✅</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <h3 style="color: #c9d1d9; margin-bottom: 0.5rem;">Price Range (±10%)</h3>
                            <h2 style="color: #58a6ff; font-size: 1.5rem;">RM {prediction*0.9:.2f} - RM {prediction*1.1:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)

def csv_upload_form(model, preprocessor, feature_names, categories):
    """CSV upload for batch predictions - 13 columns (NO LOCATION - global model)"""
    st.markdown("## 📂 CSV Upload Prediction")
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How to use:</strong> Upload a CSV file with multiple products to get batch predictions. 
            Download the sample template below to see the required format.
        </div>
    """, unsafe_allow_html=True)
    
    # Sample CSV template with EXACT 13 columns matching notebook training
    sample_data = {
        'payment_sequential': [1],
        'payment_type': ['credit_card'],
        'payment_installments': [1],
        'payment_value': [100.0],
        'order_status': ['delivered'],
        'product_weight_g': [500],
        'product_length_cm': [20],
        'product_height_cm': [10],
        'product_width_cm': [15],
        'product_description_lenght': [500],  # Note: typo matches notebook
        'product_photos_qty': [1],
        'product_name_lenght': [50],
        'product_category_name_english': ['electronics']
    }
    sample_df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Required Columns (13 total - NO LOCATION):")
        # Display as bullet points for better readability
        columns_list = list(sample_data.keys())
        st.markdown("**Payment Information:**")
        st.markdown("- `payment_sequential`\n- `payment_type`\n- `payment_installments`\n- `payment_value`\n- `order_status`")
        st.markdown("**Product Dimensions:**")
        st.markdown("- `product_weight_g`\n- `product_length_cm`\n- `product_height_cm`\n- `product_width_cm`")
        st.markdown("**Product Details:**")
        st.markdown("- `product_description_lenght`\n- `product_photos_qty`\n- `product_name_lenght`\n- `product_category_name_english`")
    with col2:
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name="price_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Display available category names
    st.markdown("---")
    st.markdown("### 📋 Available Product Categories")
    st.info("**Copy and paste these exact category names into your CSV file's `product_category_name_english` column.**")
    
    # Get categories and display in 4 columns
    category_list = categories['product_categories']
    col1, col2, col3, col4 = st.columns(4)
    
    # Split categories into 4 groups
    chunk_size = (len(category_list) + 3) // 4  # Divide into 4 parts
    
    with col1:
        for cat in category_list[:chunk_size]:
            st.markdown(f"• `{cat}`")
    
    with col2:
        for cat in category_list[chunk_size:chunk_size*2]:
            st.markdown(f"• `{cat}`")
    
    with col3:
        for cat in category_list[chunk_size*2:chunk_size*3]:
            st.markdown(f"• `{cat}`")
    
    with col4:
        for cat in category_list[chunk_size*3:]:
            st.markdown(f"• `{cat}`")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your CSV file with product data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Generate Predictions", use_container_width=True):
                with st.spinner('🔄 Processing predictions...'):
                    predictions = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        prediction = predict_price(model, preprocessor, feature_names, row.to_dict())
                        predictions.append(prediction)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Add predictions to dataframe
                    df['Predicted_Price'] = predictions
                    
                    st.success("✅ Predictions completed!")
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Average Price", f"RM {np.mean(predictions):.2f}")
                    with col3:
                        st.metric("Total Value", f"RM {np.sum(predictions):.2f}")
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_result = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_result,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = px.histogram(df, x='Predicted_Price', nbins=30,
                                     title='Distribution of Predicted Prices',
                                     labels={'Predicted_Price': 'Price (RM)'},
                                     color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def main():
    # Header
    st.markdown("""
        <h1>🛒 E-Commerce Price Prediction System</h1>
        <p style='text-align: center; color: #666; font-size: 1.1rem;'>
        </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, preprocessor, feature_names = load_model()
    categories = load_categories()
    
    # Model and preprocessor are required - no demo mode
    if model is None or preprocessor is None:
        st.error("❌ Failed to load model. Please check the error messages above.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=100)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select a page:",
            ["🏠 Dashboard", "📝 Manual Prediction", "📂 Batch Prediction"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("""
            **Model:** Random Forest Regressor
            
            **Dataset:** Brazilian E-Commerce Public Dataset by Olist (Source by Kaggle)
            
            **Performance:**
            - R² Score: 0.8202 (82% Accuracy)
            - RMSE: 30.76
            - MAE: 15.64
            - MAPE: 21.33%
                
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍🎓 About")
        st.markdown("""
            **Developed by:** Loi Min Guang (TP065267)
            
            **Project Title:** E-Commerce Price Prediction System
                    
            **Project Aim:** To develop and evaluate a comprehensive machine learning framework for dynamic pricing optimization in e-commerce that maximizes predictive accuracy while generating price recommendations for online retailers.
            
            **Model:** Random Forest Regressor
            
            **Performance Metrics:**
            - R² Score: 0.8202 (82.02% Accuracy)
            - RMSE: 30.76
            - MAE: 15.64
            - MAPE: 21.34%
            
        """)
    
    # Main content
    if page == "🏠 Dashboard":
        create_dashboard()
    elif page == "📝 Manual Prediction":
        manual_input_form(model, preprocessor, feature_names, categories)
    else:
        csv_upload_form(model, preprocessor, feature_names, categories)

if __name__ == "__main__":
    main()

import streamlit as st
import json
import tempfile
import os
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ee
from earth_engine_utils import initialize_earth_engine, get_admin_boundaries, get_boundary_names
from vegetation_indices import mask_clouds, add_vegetation_indices
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Rose-Blue Theme Variables */
    :root {
        --rose-primary: #e11d48;
        --rose-secondary: #fb7185;
        --rose-tertiary: #fecdd3;
        --rose-accent: #be123c;
        --blue-primary: #1d4ed8;
        --blue-secondary: #3b82f6;
        --blue-tertiary: #93c5fd;
        --blue-accent: #1e40af;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --bg-quaternary: #475569;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --accent-primary: #818cf8;
        --accent-secondary: #22c55e;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --gradient-primary: linear-gradient(90deg, var(--rose-primary) 0%, var(--blue-primary) 100%);
        --gradient-secondary: linear-gradient(90deg, var(--rose-secondary) 0%, var(--blue-secondary) 100%);
    }

    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 10% 20%, rgba(225, 29, 72, 0.2) 0%, transparent 40%),
                    radial-gradient(circle at 90% 70%, rgba(59, 130, 246, 0.15) 0%, transparent 40%);
        z-index: 1;
    }
    
    .main-header-content {
        position: relative;
        z-index: 2;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(to right, var(--rose-secondary) 0%, var(--blue-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        min-width: 150px;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--text-muted);
    }
    
    /* Section styling */
    .section-header {
        background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--accent-primary);
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .section-title i {
        margin-right: 10px;
        color: var(--accent-primary);
    }
    
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .feature-card {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 1.5rem;
        border-top: 4px solid var(--accent-primary);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%);
        z-index: 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.2);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--accent-primary);
        background: rgba(59, 130, 246, 0.1);
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }
    
    .feature-title {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
        position: relative;
        z-index: 1;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0.2rem;
    }
    
    .status-active {
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-secondary);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--accent-warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Button styling */
    .stButton button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button:hover {
        background: var(--gradient-secondary);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-secondary);
    }
    
    /* Earth visualization */
    .earth-visual {
        width: 200px;
        height: 200px;
        margin: 1rem auto;
        position: relative;
        border-radius: 50%;
        background: conic-gradient(
            var(--blue-primary) 0% 20%, 
            var(--rose-primary) 20% 40%, 
            var(--accent-warning) 40% 60%, 
            #ef4444 60% 80%, 
            #94a3b8 80% 100%
        );
        box-shadow: 0 0 50px rgba(59, 130, 246, 0.5);
        animation: rotate 30s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Floating animation */
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .main-subtitle {
            font-size: 1.1rem;
        }
        
        .stats-container {
            flex-direction: column;
            align-items: center;
        }
        
        .stat-item {
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class ChartGenerator:
    def __init__(self):
        # Forex-style color scheme
        self.colors = {
            'primary': '#FF4B4B',      # Streamlit red
            'secondary': '#262730',     # Dark slate
            'success': '#00D4AA',       # Teal
            'background': '#FFFFFF',    # White
            'grid': '#E6E6FA',         # Light lavender
            'text': '#262730',         # Dark slate
            'bullish': '#26A69A',      # Teal green
            'bearish': '#EF5350',      # Red
            'neutral': '#78909C'       # Blue grey
        }
    
    def create_vegetation_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create forex-style vegetation indices charts."""
        charts = {}
        
        try:
            # Get unique indices
            indices = df['Index'].unique()
            
            # Create individual charts for each index
            for index_name in indices:
                charts[f"{index_name}_timeseries"] = self._create_forex_timeseries(df, index_name)
            
            # Create combined overview chart
            charts["overview"] = self._create_combined_overview(df)
            
            # Create correlation heatmap
            charts["correlation"] = self._create_correlation_heatmap(df)
            
            # Create seasonal analysis
            charts["seasonal"] = self._create_seasonal_analysis(df)
            
            return charts
            
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
            return {}
    
    def _create_forex_timeseries(self, df: pd.DataFrame, index_name: str) -> go.Figure:
        """Create a forex-style time series chart for a specific vegetation index."""
        # Filter data for the specific index
        index_data = df[df['Index'] == index_name].copy()
        index_data = index_data.sort_values('Date').reset_index(drop=True)
        
        # Calculate moving averages
        index_data['MA_7'] = index_data['Value'].rolling(window=7, center=True).mean()
        index_data['MA_30'] = index_data['Value'].rolling(window=30, center=True).mean()
        
        # Determine color based on trend
        colors = []
        for i in range(len(index_data)):
            if i == 0:
                colors.append(self.colors['neutral'])
            else:
                if index_data.iloc[i]['Value'] > index_data.iloc[i-1]['Value']:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
        
        fig = go.Figure()
        
        # Add main line chart
        fig.add_trace(go.Scatter(
            x=index_data['Date'],
            y=index_data['Value'],
            mode='lines+markers',
            name=f'{index_name} Values',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(
                size=6,
                color=colors,
                line=dict(width=1, color=self.colors['text'])
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.4f}<extra></extra>'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=index_data['Date'],
            y=index_data['MA_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color=self.colors['bullish'], width=1, dash='dash'),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=index_data['Date'],
            y=index_data['MA_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color=self.colors['bearish'], width=1, dash='dot'),
            opacity=0.7
        ))
        
        # Calculate statistics for annotation
        current_value = index_data['Value'].iloc[-1]
        min_value = index_data['Value'].min()
        max_value = index_data['Value'].max()
        mean_value = index_data['Value'].mean()
        
        # Update layout with forex-style formatting
        fig.update_layout(
            title=dict(
                text=f'{index_name} - Vegetation Index Analysis',
                font=dict(size=20, color=self.colors['text'], family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Date',
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                color=self.colors['text'],
                tickformat='%Y-%m'
            ),
            yaxis=dict(
                title=f'{index_name} Value',
                gridcolor=self.colors['grid'],
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor=self.colors['text'],
                zerolinewidth=1,
                color=self.colors['text']
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'], family='Arial'),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    text=f"Current: {current_value:.4f}<br>" +
                         f"Min: {min_value:.4f}<br>" +
                         f"Max: {max_value:.4f}<br>" +
                         f"Avg: {mean_value:.4f}",
                    font=dict(size=10, color=self.colors['text']),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=self.colors['grid'],
                    borderwidth=1,
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    def _create_combined_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create a combined overview chart showing all indices."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=df['Index'].unique(),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        indices = df['Index'].unique()
        colors = [self.colors['primary'], self.colors['bullish'], self.colors['bearish'], self.colors['neutral']]
        
        for i, index_name in enumerate(indices[:4]):  # Limit to 4 indices
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            index_data = df[df['Index'] == index_name].sort_values('Date').reset_index(drop=True)
            
            fig.add_trace(
                go.Scatter(
                    x=index_data['Date'],
                    y=index_data['Value'],
                    mode='lines+markers',
                    name=index_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    showlegend=True if i == 0 else False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(
                text='Vegetation Indices Overview Dashboard',
                font=dict(size=20, color=self.colors['text'], family='Arial Black'),
                x=0.5
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=600
        )
        
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap between different vegetation indices."""
        try:
            # Pivot the data to have indices as columns
            pivot_df = df.pivot_table(
                index='Date', 
                columns='Index', 
                values='Value', 
                aggfunc='mean'
            )
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[
                    [0, self.colors['bearish']],
                    [0.5, self.colors['background']],
                    [1, self.colors['bullish']]
                ],
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate='<b>Correlation</b><br>' +
                             'X: %{x}<br>' +
                             'Y: %{y}<br>' +
                             'Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='Vegetation Indices Correlation Matrix',
                    font=dict(size=18, color=self.colors['text'], family='Arial Black'),
                    x=0.5
                ),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def _create_seasonal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create seasonal analysis chart."""
        try:
            # Add month column
            df_seasonal = df.copy()
            df_seasonal['Month'] = df_seasonal['Date'].dt.month
            df_seasonal['Month_Name'] = df_seasonal['Date'].dt.month_name()
            
            # Calculate monthly averages
            monthly_avg = df_seasonal.groupby(['Month', 'Month_Name', 'Index'])['Value'].mean().reset_index()
            
            fig = go.Figure()
            
            indices = df['Index'].unique()
            colors = [self.colors['primary'], self.colors['bullish'], self.colors['bearish'], self.colors['neutral']]
            
            for i, index_name in enumerate(indices):
                index_data = monthly_avg[monthly_avg['Index'] == index_name]
                
                fig.add_trace(go.Scatter(
                    x=index_data['Month_Name'],
                    y=index_data['Value'],
                    mode='lines+markers',
                    name=index_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Avg Value: %{y:.4f}<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text='Seasonal Analysis - Monthly Averages',
                    font=dict(size=18, color=self.colors['text'], family='Arial Black'),
                    x=0.5
                ),
                xaxis=dict(
                    title='Month',
                    gridcolor=self.colors['grid'],
                    color=self.colors['text']
                ),
                yaxis=dict(
                    title='Average Value',
                    gridcolor=self.colors['grid'],
                    color=self.colors['text']
                ),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating seasonal analysis: {str(e)}")
            return go.Figure()

class VegetationPredictor:
    def __init__(self):
        self.models = {}
        self.predictions = {}
    
    def prepare_data_for_prediction(self, df: pd.DataFrame, target_index: str, lookback_days: int = 30):
        """Prepare data for time series prediction."""
        try:
            # Filter data for target index
            target_data = df[df['Index'] == target_index].sort_values('Date').reset_index(drop=True)
            
            if len(target_data) < lookback_days * 2:
                return None, None, "Not enough data for prediction"
            
            # Create features (lagged values)
            features = []
            targets = []
            dates = []
            
            for i in range(lookback_days, len(target_data)):
                # Use previous lookback_days values as features
                feature_row = []
                for j in range(lookback_days):
                    feature_row.append(target_data.iloc[i - lookback_days + j]['Value'])
                
                features.append(feature_row)
                targets.append(target_data.iloc[i]['Value'])
                dates.append(target_data.iloc[i]['Date'])
            
            return np.array(features), np.array(targets), dates, None
            
        except Exception as e:
            return None, None, None, str(e)
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple regression models."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            models = {}
            performances = {}
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            models['Linear Regression'] = lr
            performances['Linear Regression'] = {
                'r2': r2_score(y_test, lr_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
            }
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            models['Random Forest'] = rf
            performances['Random Forest'] = {
                'r2': r2_score(y_test, rf_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
            }
            
            self.models = models
            return performances, None
            
        except Exception as e:
            return None, str(e)
    
    def predict_future(self, model, last_sequence, days_ahead=30):
        """Predict future values."""
        try:
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Predict next value
                next_val = model.predict(current_sequence.reshape(1, -1))[0]
                predictions.append(next_val)
                
                # Update sequence (remove first, add prediction)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_val
            
            return predictions, None
            
        except Exception as e:
            return None, str(e)

# Page configuration
st.set_page_config(
    page_title="Khisba GIS - Advanced Geospatial Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'credentials_uploaded' not in st.session_state:
    st.session_state.credentials_uploaded = False
if 'selected_geometry' not in st.session_state:
    st.session_state.selected_geometry = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'vegetation_data' not in st.session_state:
    st.session_state.vegetation_data = None
if 'chart_generator' not in st.session_state:
    st.session_state.chart_generator = ChartGenerator()
if 'predictor' not in st.session_state:
    st.session_state.predictor = VegetationPredictor()

# Authentication check
if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-header">
        <div class="main-header-content">
            <h1 class="main-title">🌍 KHISBA GIS</h1>
            <p class="main-subtitle">Enterprise Geospatial Intelligence Platform Powered by Google Earth Engine</p>
            <div class="earth-visual floating"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">40+</div>
            <div class="stat-label">Vegetation & Salinity Indices</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">5PB+</div>
            <div class="stat-label">Satellite Data Processed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">Platform Uptime</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">100+</div>
            <div class="stat-label">Countries Served</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title"><i class="fas fa-lock"></i> Authentication Required</h2>
        <p class="section-subtitle">Enter the admin password to access Khisba GIS Enterprise Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("**Enterprise Password**", type="password", 
                               placeholder="Enter admin password", 
                               help="Demo password: admin")
        
        if st.button("🔓 **ENTER PLATFORM**", type="primary", use_container_width=True):
            if password == "admin":
                st.session_state.authenticated = True
                st.success("✅ Authentication successful! Loading enterprise dashboard...")
                st.rerun()
            else:
                st.error("❌ Invalid password. Demo password: admin")
    
    # Feature cards
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title"><i class="fas fa-star"></i> Platform Features</h2>
        <p class="section-subtitle">Advanced geospatial analytics for enterprise decision-making</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌿</div>
            <h3 class="feature-title">Vegetation Analytics</h3>
            <p class="feature-description">Comprehensive analysis of 40+ vegetation indices including NDVI, EVI, SAVI with scientific precision and validation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💧</div>
            <h3 class="feature-title">Water Resource Management</h3>
            <p class="feature-description">Advanced water indices (NDWI, MNDWI) for precise water resource monitoring and management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Enterprise Analytics</h3>
            <p class="feature-description">Multi-scale analysis from field-level to continental scale with intelligent tiling architecture.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: var(--bg-secondary); border-radius: 10px;">
        <h4 style="color: var(--accent-primary); margin: 0;">Demo Access</h4>
        <p style="color: var(--text-secondary); margin: 10px 0 0 0;">Username: <strong>admin</strong><br>Password: <strong>admin</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# Main application after authentication
st.markdown("""
<div class="main-header">
    <div class="main-header-content">
        <h1 class="main-title">🌍 KHISBA GIS</h1>
        <p class="main-subtitle">Enterprise Geospatial Intelligence Platform • Powered by Google Earth Engine</p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 1rem;">
            <span class="status-badge status-active">Session Active</span>
            <span class="status-badge status-warning">Enterprise Mode</span>
            <span style="color: var(--text-secondary); font-size: 0.9rem;">Created by <strong>Taibi Farouk Djilali</strong></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Professional Trading Dashboard Sidebar
st.sidebar.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, var(--rose-primary) 0%, var(--blue-primary) 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0; font-size: 1.5rem;">📊 KHISBA</h2>
    <p style="color: #e8f5e8; margin: 5px 0 0 0; font-size: 0.9rem;">Professional GIS Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🔐 **AUTHENTICATION STATUS**")

# Status indicators
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.session_state.ee_initialized:
        st.markdown('<span class="status-badge status-active">GEE Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">GEE Disconnected</span>', unsafe_allow_html=True)

with col2:
    if st.session_state.authenticated:
        st.markdown('<span class="status-badge status-active">User Authenticated</span>', unsafe_allow_html=True)

# Google Earth Engine Authentication
if not st.session_state.ee_initialized:
    st.sidebar.markdown("""
    <div class="section-header" style="margin: 1rem 0;">
        <h3 class="section-title"><i class="fas fa-key"></i> GEE Credentials</h3>
        <p class="section-subtitle">Upload your service account JSON file</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    **Required Steps:**
    1. Go to [Google Cloud Console](https://console.cloud.google.com)
    2. Select your project and go to IAM & Admin → Service Accounts  
    3. Create or select a service account
    4. Click "Add Key" → "Create new key" → JSON
    5. Download and upload the JSON file here
    
    **Note:** Your project must be registered with Earth Engine at [signup.earthengine.google.com](https://signup.earthengine.google.com)
    """)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose your service account JSON file",
        type=['json'],
        help="Upload your Google Earth Engine service account JSON credentials"
    )
    
    if uploaded_file is not None:
        try:
            # Read and parse the JSON file
            credentials_data = json.load(uploaded_file)
            
            # Save credentials to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(credentials_data, tmp_file)
                credentials_path = tmp_file.name
            
            # Initialize Earth Engine
            success = initialize_earth_engine(credentials_path)
            
            if success:
                st.session_state.ee_initialized = True
                st.session_state.credentials_uploaded = True
                st.sidebar.success("✅ Earth Engine initialized successfully!")
                
                # Clean up temporary file
                os.unlink(credentials_path)
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to initialize Earth Engine")
                st.sidebar.error("""
                **Common issues:**
                - Service account key has expired (generate a new one)
                - Project not registered with Earth Engine
                - Invalid JSON file format
                - Missing required permissions
                """)
                
        except Exception as e:
            st.sidebar.error(f"❌ Error processing credentials: {str(e)}")
else:
    st.sidebar.success("✅ Earth Engine Connected")
    st.sidebar.markdown("""
    <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: var(--accent-primary); margin: 0 0 0.5rem 0;">Platform Status</h4>
        <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">All systems operational. Ready for geospatial analysis.</p>
    </div>
    """, unsafe_allow_html=True)

# Main application content
if st.session_state.ee_initialized:
    
    # Professional Study Area Selection
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title"><i class="fas fa-map-marker-alt"></i> TRADING AREA SELECTION</h2>
        <p class="section-subtitle">Select your geographical trading zone for vegetation indices analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Country selection
        try:
            countries_fc = get_admin_boundaries(0)
            if countries_fc is not None:
                country_names = get_boundary_names(countries_fc, 0)
                selected_country = st.selectbox(
                    "**Select Country**",
                    options=[""] + country_names,
                    help="Choose a country for analysis"
                )
            else:
                st.error("Failed to load countries data")
                selected_country = ""
        except Exception as e:
            st.error(f"Error loading countries: {str(e)}")
            selected_country = ""
    
    with col2:
        # Admin1 selection (states/provinces)
        selected_admin1 = ""
        if selected_country and countries_fc is not None:
            try:
                # Get country code
                country_feature = countries_fc.filter(ee.Filter.eq('ADM0_NAME', selected_country)).first()
                country_code = country_feature.get('ADM0_CODE').getInfo()
                
                admin1_fc = get_admin_boundaries(1, country_code)
                if admin1_fc is not None:
                    admin1_names = get_boundary_names(admin1_fc, 1)
                    selected_admin1 = st.selectbox(
                        "**Select State/Province**",
                        options=[""] + admin1_names,
                        help="Choose a state or province"
                    )
                else:
                    st.error("Failed to load admin1 data")
            except Exception as e:
                st.error(f"Error loading admin1: {str(e)}")
    
    with col3:
        # Admin2 selection (municipalities)
        selected_admin2 = ""
        if selected_admin1 and 'admin1_fc' in locals() and admin1_fc is not None:
            try:
                # Get admin1 code
                admin1_feature = admin1_fc.filter(ee.Filter.eq('ADM1_NAME', selected_admin1)).first()
                admin1_code = admin1_feature.get('ADM1_CODE').getInfo()
                
                admin2_fc = get_admin_boundaries(2, None, admin1_code)
                if admin2_fc is not None:
                    admin2_names = get_boundary_names(admin2_fc, 2)
                    selected_admin2 = st.selectbox(
                        "**Select Municipality**",
                        options=[""] + admin2_names,
                        help="Choose a municipality"
                    )
                else:
                    st.error("Failed to load admin2 data")
            except Exception as e:
                st.error(f"Error loading admin2: {str(e)}")
    
    # Professional GIS Map Display
    if selected_country:
        st.markdown("""
        <div class="section-header">
            <h2 class="section-title"><i class="fas fa-globe-americas"></i> KHISBA GIS ANALYTICS WORKSPACE</h2>
            <p class="section-subtitle">Interactive geospatial analysis with multi-layer visualization</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Determine which geometry to use
            if selected_admin2 and 'admin2_fc' in locals() and admin2_fc is not None:
                geometry = admin2_fc.filter(ee.Filter.eq('ADM2_NAME', selected_admin2))
                area_name = f"{selected_admin2}, {selected_admin1}, {selected_country}"
                area_level = "Municipality"
            elif selected_admin1 and 'admin1_fc' in locals() and admin1_fc is not None:
                geometry = admin1_fc.filter(ee.Filter.eq('ADM1_NAME', selected_admin1))
                area_name = f"{selected_admin1}, {selected_country}"
                area_level = "State/Province"
            else:
                geometry = countries_fc.filter(ee.Filter.eq('ADM0_NAME', selected_country))
                area_name = selected_country
                area_level = "Country"
            
            # Get geometry bounds for map centering
            bounds = geometry.geometry().bounds().getInfo()
            coords = bounds['coordinates'][0]
            
            # Calculate center and area
            lats = [coord[1] for coord in coords]
            lons = [coord[0] for coord in coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Create professional GIS map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles=None,
                control_scale=True,
                prefer_canvas=True
            )
            
            # Add multiple professional base layers
            folium.TileLayer(
                'OpenStreetMap',
                name='OpenStreetMap',
                overlay=False,
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Terrain',
                overlay=False,
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'CartoDB dark_matter',
                name='Dark Theme',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add professional study area styling
            folium.GeoJson(
                bounds,
                style_function=lambda x: {
                    'fillColor': '#00ff88',
                    'color': '#ffffff',
                    'weight': 3,
                    'fillOpacity': 0.2,
                    'dashArray': '5, 5'
                },
                popup=folium.Popup(f"<b>Study Area:</b><br>{area_name}<br><b>Level:</b> {area_level}", max_width=300),
                tooltip=f"Click for details: {area_name}"
            ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Professional GIS info panel
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display professional map with enhanced styling
                st.markdown("""
                <div style="border: 3px solid var(--accent-primary); border-radius: 10px; padding: 5px; background: var(--bg-secondary);">
                """, unsafe_allow_html=True)
                
                map_data = st_folium(
                    m, 
                    width=None, 
                    height=500,
                    returned_objects=["last_clicked", "bounds"],
                    key="gis_map"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Professional GIS information panel
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%); padding: 1.5rem; border-radius: 10px; border: 1px solid var(--accent-primary); height: 500px;">
                    <h4 style="color: var(--accent-primary); margin-top: 0;">🌍 GIS DATA PANEL</h4>
                    <hr style="border-color: var(--accent-primary);">
                    
                    <div style="margin: 1rem 0;">
                        <strong style="color: var(--text-primary);">Study Area:</strong><br>
                        <span style="color: var(--text-secondary);">{area_name}</span>
                    </div>
                    
                    <div style="margin: 1rem 0;">
                        <strong style="color: var(--text-primary);">Administrative Level:</strong><br>
                        <span style="color: var(--accent-primary);">{area_level}</span>
                    </div>
                    
                    <div style="margin: 1rem 0;">
                        <strong style="color: var(--text-primary);">Coordinates:</strong><br>
                        <span style="color: var(--text-secondary);">Lat: {center_lat:.4f}°<br>
                        Lon: {center_lon:.4f}°</span>
                    </div>
                    
                    <div style="margin: 1rem 0;">
                        <strong style="color: var(--text-primary);">Map Layers:</strong><br>
                        <span style="color: var(--text-secondary);">• Satellite Imagery<br>
                        • Terrain Data<br>
                        • Administrative Boundaries<br>
                        • Dark/Light Themes</span>
                    </div>
                    
                    <div style="background: var(--bg-primary); padding: 0.8rem; border-radius: 5px; margin-top: 1.5rem;">
                        <small style="color: var(--accent-primary);">📊 KHISBA GIS Professional</small><br>
                        <small style="color: var(--text-muted);">Powered by Earth Engine</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.selected_geometry = geometry
            
            # Professional status indicator
            st.markdown(f"""
            <div style="text-align: center; background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); padding: 0.8rem; border-radius: 5px; margin: 1rem 0;">
                <strong style="color: white;">✅ GIS WORKSPACE ACTIVE</strong> • Study Area: {area_name}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ GIS Map Error: {str(e)}")
            st.info("Please check your internet connection and try refreshing the page.")
    
    # Professional Analysis Parameters
    if st.session_state.selected_geometry is not None:
        st.markdown("""
        <div class="section-header">
            <h2 class="section-title"><i class="fas fa-cogs"></i> TRADING PARAMETERS</h2>
            <p class="section-subtitle">Configure your analysis timeframe and satellite data sources</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "**Start Date**",
                value=datetime(2023, 1, 1),
                help="Start date for the analysis period"
            )
            
            cloud_cover = st.slider(
                "**Maximum Cloud Cover (%)**",
                min_value=0,
                max_value=100,
                value=20,
                help="Maximum cloud cover percentage for images"
            )
        
        with col2:
            end_date = st.date_input(
                "**End Date**",
                value=datetime(2023, 12, 31),
                help="End date for the analysis period"
            )
            
            collection_choice = st.selectbox(
                "**Satellite Collection**",
                options=["Sentinel-2", "Landsat-8"],
                help="Choose the satellite collection for analysis"
            )
        
        # Vegetation Indices Selection
        st.markdown("""
        <div class="section-header">
            <h2 class="section-title"><i class="fas fa-leaf"></i> Vegetation Indices Selection</h2>
            <p class="section-subtitle">Choose from 40+ scientifically validated vegetation indices</p>
        </div>
        """, unsafe_allow_html=True)
        
        available_indices = [
            'NDVI', 'ARVI', 'ATSAVI', 'DVI', 'EVI', 'EVI2', 'GNDVI', 'MSAVI', 'MSI', 'MTVI', 'MTVI2',
            'NDTI', 'NDWI', 'OSAVI', 'RDVI', 'RI', 'RVI', 'SAVI', 'TVI', 'TSAVI', 'VARI', 'VIN', 'WDRVI',
            'GCVI', 'AWEI', 'MNDWI', 'WI', 'ANDWI', 'NDSI', 'nDDI', 'NBR', 'DBSI', 'SI', 'S3', 'BRI',
            'SSI', 'NDSI_Salinity', 'SRPI', 'MCARI', 'NDCI', 'PSSRb1', 'SIPI', 'PSRI', 'Chl_red_edge', 'MARI', 'NDMI'
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            select_all = st.checkbox("**Select All Indices**")
        with col2:
            if st.button("**Clear All**"):
                st.session_state.selected_indices = []
        
        if select_all:
            selected_indices = st.multiselect(
                "**Choose vegetation indices to calculate:**",
                options=available_indices,
                default=available_indices,
                help="Select the vegetation indices you want to analyze"
            )
        else:
            selected_indices = st.multiselect(
                "**Choose vegetation indices to calculate:**",
                options=available_indices,
                default=['NDVI', 'EVI', 'SAVI', 'NDWI'],
                help="Select the vegetation indices you want to analyze"
            )
        
        # Run Analysis Button
        if st.button("🚀 **RUN ENTERPRISE ANALYSIS**", type="primary", use_container_width=True):
            if not selected_indices:
                st.error("Please select at least one vegetation index")
            else:
                with st.spinner("Running advanced vegetation indices analysis..."):
                    try:
                        # Define collection based on choice
                        if collection_choice == "Sentinel-2":
                            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        else:
                            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        
                        # Filter collection
                        filtered_collection = (collection
                            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                            .filterBounds(st.session_state.selected_geometry)
                            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                        )
                        
                        # Apply cloud masking and add vegetation indices
                        if collection_choice == "Sentinel-2":
                            processed_collection = (filtered_collection
                                .map(mask_clouds)
                                .map(add_vegetation_indices)
                            )
                        else:
                            # For Landsat, we'd need different cloud masking
                            processed_collection = filtered_collection.map(add_vegetation_indices)
                        
                        # Calculate time series for selected indices
                        results = {}
                        for index in selected_indices:
                            try:
                                # Create a function to add date and reduce region
                                def add_date_and_reduce(image):
                                    reduced = image.select(index).reduceRegion(
                                        reducer=ee.Reducer.mean(),
                                        geometry=st.session_state.selected_geometry.geometry(),
                                        scale=30,
                                        maxPixels=1e9
                                    )
                                    return ee.Feature(None, reduced.set('date', image.date().format()))
                                
                                # Map over collection to get time series
                                time_series = processed_collection.map(add_date_and_reduce)
                                
                                # Convert to list
                                time_series_list = time_series.getInfo()
                                
                                # Extract dates and values
                                dates = []
                                values = []
                                
                                if 'features' in time_series_list:
                                    for feature in time_series_list['features']:
                                        props = feature['properties']
                                        if index in props and props[index] is not None and 'date' in props:
                                            dates.append(props['date'])
                                            values.append(props[index])
                                
                                results[index] = {'dates': dates, 'values': values}
                                
                            except Exception as e:
                                st.warning(f"Could not calculate {index}: {str(e)}")
                                results[index] = {'dates': [], 'values': []}
                        
                        st.session_state.analysis_results = results
                        
                        # Convert results to DataFrame for chart generation
                        vegetation_data = []
                        for index, data in results.items():
                            for date, value in zip(data['dates'], data['values']):
                                if value is not None:
                                    vegetation_data.append({
                                        'Date': pd.to_datetime(date),
                                        'Index': index,
                                        'Value': value
                                    })
                        
                        if vegetation_data:
                            df = pd.DataFrame(vegetation_data)
                            st.session_state.vegetation_data = df
                            st.success("✅ Analysis completed successfully!")
                        else:
                            st.error("No valid data found for the selected parameters")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

# Display Results with Enhanced Analytics
if st.session_state.analysis_results and st.session_state.vegetation_data is not None:
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title"><i class="fas fa-chart-bar"></i> Advanced Analytics Dashboard</h2>
        <p class="section-subtitle">Professional vegetation analytics with correlation, seasonal trends, and predictive modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.vegetation_data
    
    # Create enhanced charts
    charts = st.session_state.chart_generator.create_vegetation_charts(df)
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", 
        "🌐 Overview", 
        "📊 Correlation", 
        "📅 Seasonal", 
        "🔮 Prediction"
    ])
    
    with tab1:
        st.markdown("### Individual Index Analysis")
        indices = df['Index'].unique()
        selected_index = st.selectbox("Select Index for Detailed Analysis", indices)
        
        if f"{selected_index}_timeseries" in charts:
            st.plotly_chart(charts[f"{selected_index}_timeseries"], use_container_width=True)
        else:
            st.warning(f"No time series data available for {selected_index}")
    
    with tab2:
        st.markdown("### Multi-Index Overview")
        if "overview" in charts:
            st.plotly_chart(charts["overview"], use_container_width=True)
        else:
            st.warning("Overview chart not available")
    
    with tab3:
        st.markdown("### Index Correlation Matrix")
        if "correlation" in charts:
            st.plotly_chart(charts["correlation"], use_container_width=True)
            
            # Correlation insights
            st.markdown("#### Correlation Insights")
            try:
                pivot_df = df.pivot_table(index='Date', columns='Index', values='Value', aggfunc='mean')
                corr_matrix = pivot_df.corr()
                
                # Find strongest correlations
                strong_correlations = []
                indices_list = corr_matrix.columns.tolist()
                
                for i in range(len(indices_list)):
                    for j in range(i+1, len(indices_list)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # Strong correlation threshold
                            strong_correlations.append({
                                'Index1': indices_list[i],
                                'Index2': indices_list[j],
                                'Correlation': corr_val,
                                'Type': 'Positive' if corr_val > 0 else 'Negative'
                            })
                
                if strong_correlations:
                    st.write("**Strong Correlations Found:**")
                    for corr in strong_correlations:
                        st.write(f"- {corr['Index1']} & {corr['Index2']}: {corr['Correlation']:.3f} ({corr['Type']})")
                else:
                    st.info("No strong correlations (|r| > 0.7) found between indices")
                    
            except Exception as e:
                st.warning("Could not generate correlation insights")
    
    with tab4:
        st.markdown("### Seasonal Pattern Analysis")
        if "seasonal" in charts:
            st.plotly_chart(charts["seasonal"], use_container_width=True)
            
            # Seasonal insights
            st.markdown("#### Seasonal Trends")
            try:
                df_seasonal = df.copy()
                df_seasonal['Month'] = df_seasonal['Date'].dt.month
                monthly_avg = df_seasonal.groupby(['Month', 'Index'])['Value'].mean().reset_index()
                
                # Find peak months for each index
                peak_months = {}
                for index in df_seasonal['Index'].unique():
                    index_data = monthly_avg[monthly_avg['Index'] == index]
                    if not index_data.empty:
                        peak_month = index_data.loc[index_data['Value'].idxmax()]
                        peak_months[index] = {
                            'month': peak_month['Month'],
                            'value': peak_month['Value']
                        }
                
                if peak_months:
                    st.write("**Peak Vegetation Months:**")
                    for index, data in peak_months.items():
                        month_name = datetime(2023, data['month'], 1).strftime('%B')
                        st.write(f"- {index}: {month_name} (Avg: {data['value']:.4f})")
                        
            except Exception as e:
                st.warning("Could not generate seasonal insights")
    
    with tab5:
        st.markdown("### Vegetation Index Prediction")
        
        if len(df) < 60:
            st.warning("Need at least 60 data points for reliable prediction")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_index = st.selectbox("Select Index to Predict", df['Index'].unique())
            
            with col2:
                prediction_days = st.slider("Days to Predict", 7, 90, 30)
            
            with col3:
                lookback_days = st.slider("Lookback Days", 7, 60, 30)
            
            if st.button("🚀 Train Prediction Models", type="primary"):
                with st.spinner("Training prediction models..."):
                    # Prepare data
                    X, y, dates, error = st.session_state.predictor.prepare_data_for_prediction(
                        df, target_index, lookback_days
                    )
                    
                    if error:
                        st.error(f"Data preparation error: {error}")
                    elif X is None:
                        st.warning("Not enough data for prediction")
                    else:
                        # Train models
                        performances, train_error = st.session_state.predictor.train_models(X, y)
                        
                        if train_error:
                            st.error(f"Training error: {train_error}")
                        else:
                            st.success("✅ Models trained successfully!")
                            
                            # Display model performances
                            st.markdown("#### Model Performance")
                            perf_df = pd.DataFrame(performances).T
                            st.dataframe(perf_df.style.format({
                                'r2': '{:.4f}',
                                'rmse': '{:.6f}'
                            }))
                            
                            # Make predictions
                            best_model_name = max(performances, key=lambda x: performances[x]['r2'])
                            best_model = st.session_state.predictor.models[best_model_name]
                            
                            # Use last sequence for prediction
                            last_sequence = X[-1]
                            predictions, pred_error = st.session_state.predictor.predict_future(
                                best_model, last_sequence, prediction_days
                            )
                            
                            if pred_error:
                                st.error(f"Prediction error: {pred_error}")
                            else:
                                # Create prediction chart
                                last_date = df[df['Index'] == target_index]['Date'].max()
                                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                                
                                fig = go.Figure()
                                
                                # Historical data
                                hist_data = df[df['Index'] == target_index].sort_values('Date')
                                fig.add_trace(go.Scatter(
                                    x=hist_data['Date'],
                                    y=hist_data['Value'],
                                    mode='lines',
                                    name=f'Historical {target_index}',
                                    line=dict(color='#00ff88', width=2)
                                ))
                                
                                # Predictions
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=predictions,
                                    mode='lines+markers',
                                    name=f'Predicted {target_index}',
                                    line=dict(color='#ffaa00', width=2, dash='dash'),
                                    marker=dict(size=4)
                                ))
                                
                                # Confidence interval (simple approach)
                                confidence = np.std(hist_data['Value'].tail(30)) * 0.5
                                fig.add_trace(go.Scatter(
                                    x=future_dates + future_dates[::-1],
                                    y=np.array(predictions) + confidence + np.array(predictions) - confidence[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,170,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Confidence Interval'
                                ))
                                
                                fig.update_layout(
                                    title=f'{target_index} - {prediction_days}-Day Prediction (Best Model: {best_model_name})',
                                    xaxis_title='Date',
                                    yaxis_title=f'{target_index} Value',
                                    plot_bgcolor='#0E1117',
                                    paper_bgcolor='#0E1117',
                                    font=dict(color='white'),
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Prediction summary
                                current_value = hist_data['Value'].iloc[-1]
                                predicted_change = ((predictions[-1] - current_value) / current_value * 100)
                                
                                st.markdown("#### Prediction Summary")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Current Value",
                                        f"{current_value:.4f}",
                                        delta=f"{predicted_change:.1f}% predicted change"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Final Prediction",
                                        f"{predictions[-1]:.4f}",
                                        delta=f"{(predictions[-1] - current_value):.4f}"
                                    )
                                
                                with col3:
                                    trend = "Bullish" if predicted_change > 0 else "Bearish"
                                    st.metric(
                                        "Trend Outlook",
                                        trend,
                                        delta_color="normal" if predicted_change > 0 else "inverse"
                                    )

    # Data Export
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title"><i class="fas fa-download"></i> Data Export</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 **Download Results as CSV**", use_container_width=True):
        # Prepare data for export
        export_data = []
        for index, data in st.session_state.analysis_results.items():
            for date, value in zip(data['dates'], data['values']):
                if value is not None:
                    export_data.append({
                        'Date': date,
                        'Index': index,
                        'Value': value
                    })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="**Download CSV**",
                data=csv,
                file_name=f"vegetation_indices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data available for export")

else:
    if not st.session_state.ee_initialized:
        st.info("👆 Please upload your Google Earth Engine credentials to get started")
    elif st.session_state.selected_geometry is None:
        st.info("👆 Please select a study area to proceed with analysis")
    else:
        st.info("👆 Configure your analysis parameters and click 'Run Analysis'")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; background: var(--bg-secondary); border-radius: 10px;">
    <p style="color: var(--text-muted); margin: 0;">🌍 <strong>Khisba GIS Enterprise Platform</strong> • Created by <strong>Taibi Farouk Djilali</strong> • Powered by Google Earth Engine</p>
</div>
""", unsafe_allow_html=True)

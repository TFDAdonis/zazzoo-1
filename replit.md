# Overview

Khisba GIS is a professional vegetation analysis application built with Streamlit that provides comprehensive satellite-based vegetation monitoring capabilities. The application leverages Google Earth Engine for satellite data processing and offers multiple vegetation indices calculations for agricultural and environmental analysis. It features an interactive dashboard with mapping capabilities, time-series analysis, and professional trading-style visualizations for vegetation data.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web application with a professional trading dashboard aesthetic
- **UI Components**: Interactive maps using Folium and streamlit-folium integration, data visualizations with Plotly and Matplotlib
- **Layout**: Wide layout with expandable sidebar for controls and authentication
- **Styling**: Custom CSS with gradient backgrounds and professional color schemes

## Backend Architecture
- **Core Engine**: Google Earth Engine (GEE) integration for satellite data processing
- **Data Processing**: Modular approach with separate utilities for Earth Engine operations and vegetation calculations
- **Session Management**: Streamlit session state for maintaining user authentication and analysis results
- **File Structure**:
  - `app.py`: Main application interface and orchestration
  - `earth_engine_utils.py`: Earth Engine initialization and administrative boundary utilities
  - `vegetation_indices.py`: Vegetation index calculations and cloud masking

## Data Processing Pipeline
- **Satellite Data**: Sentinel-2 imagery with cloud masking capabilities
- **Administrative Boundaries**: FAO GAUL dataset for country, state, and municipal level boundaries
- **Vegetation Indices**: Multiple indices including NDVI, ARVI, ATSAVI, DVI, EVI, and EVI2
- **Temporal Analysis**: Time-series processing for vegetation trend analysis

## Authentication System
- **Method**: Google Earth Engine service account authentication
- **Credentials**: JSON-based service account key validation
- **Security**: Temporary file handling for uploaded credentials with validation checks

# External Dependencies

## Core Services
- **Google Earth Engine**: Primary satellite data platform requiring service account authentication
- **FAO GAUL Dataset**: Administrative boundary data from Food and Agriculture Organization

## Python Libraries
- **Streamlit**: Web application framework and UI components
- **Folium**: Interactive mapping and geospatial visualization
- **Plotly**: Advanced charting and interactive visualizations
- **Matplotlib**: Statistical plotting and chart generation
- **Pandas**: Data manipulation and analysis
- **Earth Engine Python API**: Google Earth Engine client library

## Data Sources
- **Sentinel-2**: European Space Agency satellite imagery
- **Administrative Boundaries**: FAO Global Administrative Unit Layers (GAUL) 2015
- **Quality Masks**: Sentinel-2 QA60 band for cloud and cirrus detection

## Visualization Stack
- **streamlit-folium**: Streamlit-Folium integration for map display
- **Plotly Express**: Simplified plotting interface
- **Plotly Graph Objects**: Advanced chart customization
- **Matplotlib Animation**: Time-series visualization capabilities
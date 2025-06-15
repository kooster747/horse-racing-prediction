"""
Horse Racing Prediction Web Application

This is the main entry point for the Flask web application that provides
horse racing predictions with real-time data updates.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

# Import the real-time data handler
sys.path.append('/home/ubuntu/horse_racing_prediction')
try:
    from real_time_handler import RealTimeDataHandler
except ImportError:
    print("Warning: real_time_handler module not found. Real-time updates will be simulated.")
    RealTimeDataHandler = None

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'horse_racing_prediction_secret_key'

# Enable database
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'mydb')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Set paths
MODEL_DIR = '/home/ubuntu/horse_racing_prediction/models'
DATA_DIR = '/home/ubuntu/horse_racing_prediction/data'
RESULTS_DIR = '/home/ubuntu/horse_racing_prediction/results'
TEST_DIR = '/home/ubuntu/horse_racing_prediction/test_results'

# Load the prediction model
def load_model():
    try:
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load model metadata
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Model '{metadata['model_name']}' loaded successfully.")
        return model, metadata
    except FileNotFoundError as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Initialize real-time data handler
def init_real_time_handler():
    if RealTimeDataHandler:
        handler = RealTimeDataHandler(DATA_DIR)
        
        # Configure data sources (placeholder for real implementation)
        handler.configure_data_sources({
            'source1': {
                'name': 'Racing API',
                'base_url': 'https://api.racing-example.com/v1',
                'upcoming_endpoint': '/races/upcoming',
                'requires_auth': True,
                'auth_type': 'api_key',
                'enabled': True
            },
            'source2': {
                'name': 'Odds Provider',
                'base_url': 'https://odds-example.com/api',
                'upcoming_endpoint': '/events/horse-racing',
                'requires_auth': True,
                'auth_type': 'oauth',
                'enabled': True
            }
        })
        
        # Start monitoring with 30-second updates
        handler.start_monitoring(update_interval=30)
        return handler
    else:
        return None

# Load optimal betting strategy
def load_betting_strategy():
    try:
        strategy_path = os.path.join(TEST_DIR, 'betting_strategies.json')
        with open(strategy_path, 'r') as f:
            strategies = json.load(f)
        return strategies['strategies']['optimal']
    except FileNotFoundError:
        # Default strategy if file not found
        return {
            'threshold': 0.6,
            'min_odds': 1.5,
            'max_odds': float('inf'),
            'min_ev': 0
        }

# Generate synthetic race data for demonstration
def generate_demo_races(num_races=10):
    countries = ['UK', 'US', 'FR', 'AU', 'JP', 'HK', 'AE']
    tracks = {
        'UK': ['Ascot', 'Epsom', 'Newmarket', 'Cheltenham', 'Doncaster'],
        'US': ['Churchill Downs', 'Belmont Park', 'Santa Anita', 'Saratoga', 'Pimlico'],
        'FR': ['Longchamp', 'Chantilly', 'Deauville', 'Saint-Cloud', 'Auteuil'],
        'AU': ['Flemington', 'Randwick', 'Caulfield', 'Rosehill', 'Eagle Farm'],
        'JP': ['Tokyo', 'Nakayama', 'Hanshin', 'Kyoto', 'Sapporo'],
        'HK': ['Sha Tin', 'Happy Valley'],
        'AE': ['Meydan', 'Jebel Ali']
    }
    race_classes = ['G1', 'G2', 'G3', 'Listed', 'Handicap', 'Maiden']
    going_conditions = ['Firm', 'Good', 'Good to Soft', 'Soft', 'Heavy']
    
    races = []
    now = datetime.now()
    
    for i in range(num_races):
        # Random race time in the next 24 hours
        race_time = now + timedelta(hours=np.random.randint(1, 24))
        
        # Select country and track
        country = np.random.choice(countries)
        track = np.random.choice(tracks[country])
        
        # Create race
        race = {
            'race_id': i + 1,
            'country': country,
            'track': track,
            'race_time': race_time.strftime('%Y-%m-%d %H:%M'),
            'race_name': f"{track} {'Stakes' if np.random.random() > 0.5 else 'Cup'} {np.random.randint(1, 5)}",
            'race_class': np.random.choice(race_classes),
            'distance': np.random.choice([1000, 1200, 1400, 1600, 1800, 2000, 2400, 3000, 3200]),
            'going': np.random.choice(going_conditions),
            'num_runners': np.random.randint(6, 16),
            'status': 'upcoming'
        }
        
        # Generate horses for this race
        horses = []
        for j in range(race['num_runners']):
            horse = {
                'horse_id': j + 1,
                'horse_name': f"Horse_{np.random.randint(1, 500)}",
                'age': np.random.choice([2, 3, 4, 5, 6, 7, 8], p=[0.1, 0.25, 0.25, 0.2, 0.1, 0.05, 0.05]),
                'jockey': f"Jockey_{np.random.randint(1, 100)}",
                'trainer': f"Trainer_{np.random.randint(1, 200)}",
                'weight': np.random.normal(500, 30),
                'draw': j + 1,
                'odds': np.random.exponential(10) + 1.5,
                'previous_wins': np.random.randint(0, 10),
                'win_probability': np.random.beta(2, 5),
                'expected_value': np.random.normal(0, 0.2)
            }
            
            # Apply optimal strategy to determine recommendation
            horse['recommended'] = (
                horse['win_probability'] >= 0.6 and
                horse['odds'] >= 1.5 and
                horse['expected_value'] >= 0
            )
            
            horses.append(horse)
        
        # Sort horses by win probability
        horses = sorted(horses, key=lambda x: x['win_probability'], reverse=True)
        
        # Add horses to race
        race['horses'] = horses
        races.append(race)
    
    # Sort races by time
    races = sorted(races, key=lambda x: x['race_time'])
    
    return races

# Initialize app
model, metadata = load_model()
real_time_handler = init_real_time_handler()
optimal_strategy = load_betting_strategy()

# Import routes
from src.routes.main_routes import main_bp
app.register_blueprint(main_bp)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

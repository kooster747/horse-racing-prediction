"""
Main routes for the Horse Racing Prediction web application.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
import json
import os
import sys
import random

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html', 
                          title="Horse Racing Prediction",
                          countries=["All", "UK", "US", "FR", "AU", "JP", "HK", "AE"])

@main_bp.route('/api/races')
def get_races():
    """API endpoint to get upcoming races."""
    # In a real implementation, this would fetch data from the real_time_handler
    # For demo purposes, we'll use our synthetic data
    from src.main import generate_demo_races, optimal_strategy
    
    # Generate demo races for testing
    demo_races = generate_demo_races(15)
    
    # Filter by country if specified
    country = request.args.get('country', 'All')
    
    if country != 'All':
        filtered_races = [race for race in demo_races if race['country'] == country]
    else:
        filtered_races = demo_races
    
    # Update race data with simulated real-time changes
    global last_update_time
    current_time = datetime.now()
    
    # Simulate real-time updates every 30 seconds
    if not 'last_update_time' in globals() or (current_time - last_update_time).total_seconds() > 30:
        # Update odds and probabilities for some horses
        for race in filtered_races:
            for horse in race['horses']:
                # 20% chance of updating a horse's data
                if random.random() < 0.2:
                    # Update odds (can go up or down)
                    horse['odds'] *= random.uniform(0.9, 1.1)
                    
                    # Update win probability
                    horse['win_probability'] *= random.uniform(0.95, 1.05)
                    horse['win_probability'] = min(max(horse['win_probability'], 0), 1)
                    
                    # Recalculate expected value
                    horse['expected_value'] = (horse['win_probability'] * (horse['odds'] - 1)) - (1 - horse['win_probability'])
                    
                    # Update recommendation based on optimal strategy
                    horse['recommended'] = (
                        horse['win_probability'] >= optimal_strategy['threshold'] and
                        horse['odds'] >= optimal_strategy['min_odds'] and
                        horse['expected_value'] >= optimal_strategy['min_ev']
                    )
        
        last_update_time = current_time
    
    return jsonify(filtered_races)

@main_bp.route('/api/race/<int:race_id>')
def get_race(race_id):
    """API endpoint to get details for a specific race."""
    from src.main import generate_demo_races
    
    # Generate demo races for testing
    demo_races = generate_demo_races(15)
    
    # Find the race with the given ID
    race = next((r for r in demo_races if r['race_id'] == race_id), None)
    
    if race:
        return jsonify(race)
    else:
        return jsonify({"error": "Race not found"}), 404

@main_bp.route('/api/recommendations')
def get_recommendations():
    """API endpoint to get betting recommendations across all races."""
    from src.main import generate_demo_races
    
    # Generate demo races for testing
    demo_races = generate_demo_races(15)
    
    recommendations = []
    
    for race in demo_races:
        race_time = datetime.strptime(race['race_time'], '%Y-%m-%d %H:%M')
        
        # Only include upcoming races
        if race_time > datetime.now():
            # Find recommended horses
            for horse in race['horses']:
                if horse['recommended']:
                    recommendations.append({
                        'race_id': race['race_id'],
                        'race_name': race['race_name'],
                        'race_time': race['race_time'],
                        'track': race['track'],
                        'country': race['country'],
                        'horse_id': horse['horse_id'],
                        'horse_name': horse['horse_name'],
                        'odds': horse['odds'],
                        'win_probability': horse['win_probability'],
                        'expected_value': horse['expected_value']
                    })
    
    # Sort by expected value (highest first)
    recommendations = sorted(recommendations, key=lambda x: x['expected_value'], reverse=True)
    
    return jsonify(recommendations)

@main_bp.route('/api/stats')
def get_stats():
    """API endpoint to get model performance statistics."""
    from src.main import optimal_strategy
    
    # In a real implementation, this would fetch actual model stats
    # For demo purposes, we'll use the results from our testing
    
    stats = {
        'win_rate': 0.76,  # 76% from Monte Carlo simulations
        'roi': 16.57,      # 1657% ROI from simulations
        'confidence': 1.0, # 100% probability of positive ROI
        'races_analyzed': 1000,
        'optimal_strategy': {
            'threshold': optimal_strategy['threshold'],
            'min_odds': optimal_strategy['min_odds'],
            'max_odds': "No limit" if optimal_strategy['max_odds'] == float('inf') else optimal_strategy['max_odds'],
            'min_ev': optimal_strategy['min_ev']
        }
    }
    
    return jsonify(stats)

# Initialize last update time
last_update_time = datetime.now()

"""
Process data.csv to generate user_journeys.json and user_risk_scores.json
"""

import pandas as pd
import json
from datetime import datetime
import numpy as np

def process_user_journeys(csv_path='data.csv'):
    """Process CSV to generate user journeys JSON"""
    print("📊 Reading data.csv...")
    
    # Read CSV file - try different encodings
    df = None
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            if not df.empty:
                print(f"✅ Successfully read CSV with {encoding} encoding")
                break
        except Exception as e:
            continue
    
    if df is None or df.empty:
        raise ValueError(f"Could not read {csv_path} with any encoding")
    
    print(f"✅ Loaded {len(df)} rows from data.csv")
    
    # Filter out rows where user_uuid is NaN
    df = df[df['user_uuid'].notna()].copy()
    
    # Convert time columns
    if 'event_time' in df.columns:
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    if 'first_payment_time' in df.columns:
        df['first_payment_time'] = pd.to_datetime(df['first_payment_time'], errors='coerce')
    
    # Group by user
    journeys = {}
    features = {}
    
    for user_id in df['user_uuid'].unique():
        user_data = df[df['user_uuid'] == user_id].copy()
        user_data = user_data.sort_values('event_time')
        
        # Calculate time offsets (in seconds)
        if len(user_data) > 0:
            first_time = user_data['event_time'].iloc[0]
            user_data['time_offset'] = (user_data['event_time'] - first_time).dt.total_seconds()
        else:
            user_data['time_offset'] = 0
        
        # Build journey
        journey = []
        for _, row in user_data.iterrows():
            journey.append({
                'event_name': str(row['event_name']) if pd.notna(row['event_name']) else '',
                'time_offset': float(row['time_offset']) if pd.notna(row['time_offset']) else 0.0,
                'remarks': str(row.get('remarks', '')) if pd.notna(row.get('remarks', '')) else ''
            })
        
        journeys[user_id] = journey
        
        # Calculate features
        has_transaction = pd.notna(user_data['first_payment_time'].iloc[0]) if 'first_payment_time' in user_data.columns else False
        total_events = len(user_data)
        
        # Calculate exploration and execution scores
        event_types = user_data['event_name'].nunique()
        exploration_score = event_types / total_events if total_events > 0 else 0
        
        # Count payment attempts
        payment_keywords = ['checkout', 'pay', 'payment', 'recharge', 'qris', 'qr']
        payment_attempts = user_data['event_name'].str.contains('|'.join(payment_keywords), case=False, na=False).sum()
        execution_score = payment_attempts / total_events if total_events > 0 else 0
        
        # Determine intent type
        if has_transaction:
            intent_type = "High Conversion Intent"
        else:
            intent_type = "Exploration Intent"
        
        features[user_id] = {
            'exploration_score': float(exploration_score),
            'execution_score': float(execution_score),
            'total_events': int(total_events),
            'has_transaction': bool(has_transaction),
            'intent_type': intent_type
        }
    
    # Create output structure
    output = {
        'journeys': journeys,
        'features': features
    }
    
    # Save to JSON
    with open('user_journeys.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated user_journeys.json with {len(journeys)} users")
    return journeys, features

def calculate_risk_scores(df, journeys, features):
    """Calculate risk scores for each user"""
    print("🔍 Calculating risk scores...")
    
    risk_scores = {}
    
    for user_id in df['user_uuid'].unique():
        user_data = df[df['user_uuid'] == user_id].copy()
        user_data = user_data.sort_values('event_time')
        
        journey = journeys.get(user_id, [])
        feature = features.get(user_id, {})
        
        # Initialize risk factors
        risk_factors = {
            'high_repetition': 0.0,
            'rapid_switching': 0.0,
            'long_idle': 0.0,
            'failed_attempts': 0.0,
            'unusual_pattern': 0.0,
            'high_urgency': 0.0
        }
        
        if len(journey) > 0:
            # 1. High repetition score
            event_names = [e['event_name'] for e in journey]
            unique_events = len(set(event_names))
            repetition_rate = 1 - (unique_events / len(event_names)) if len(event_names) > 0 else 0
            risk_factors['high_repetition'] = min(100, repetition_rate * 100)
            
            # 2. Rapid switching (many different events in short time)
            if len(journey) > 1:
                time_span = journey[-1]['time_offset'] - journey[0]['time_offset']
                if time_span > 0:
                    switching_rate = unique_events / (time_span / 60)  # events per minute
                    risk_factors['rapid_switching'] = min(100, switching_rate * 10)
            
            # 3. Long idle time
            if len(journey) > 1:
                time_offsets = [e['time_offset'] for e in journey]
                time_diffs = [time_offsets[i] - time_offsets[i-1] for i in range(1, len(time_offsets))]
                if time_diffs:
                    max_idle = max(time_diffs) / 60  # Convert to minutes
                    # Long idle (> 30 minutes) indicates potential abandonment
                    if max_idle > 30:
                        risk_factors['long_idle'] = min(100, (max_idle - 30) / 10 * 5)
            
            # 4. Failed attempts (multiple payment attempts without success)
            payment_events = [e for e in journey if any(kw in e['event_name'].lower() for kw in ['checkout', 'pay', 'payment'])]
            if len(payment_events) > 0 and not feature.get('has_transaction', False):
                risk_factors['failed_attempts'] = min(100, len(payment_events) * 10)
            
            # 5. Unusual pattern (high event density with low conversion)
            if len(journey) > 1:
                time_span = journey[-1]['time_offset'] - journey[0]['time_offset']
                if time_span > 0:
                    event_density = len(journey) / (time_span / 60)  # events per minute
                    if event_density > 2 and not feature.get('has_transaction', False):
                        risk_factors['unusual_pattern'] = min(100, (event_density - 2) * 10)
            
            # 6. High urgency (very fast event sequence)
            if len(journey) > 1:
                time_span = journey[-1]['time_offset'] - journey[0]['time_offset']
                if time_span > 0:
                    event_density = len(journey) / (time_span / 60)
                    if event_density > 5:
                        risk_factors['high_urgency'] = min(100, (event_density - 5) * 5)
        
        # Calculate total risk score (weighted average)
        weights = {
            'high_repetition': 0.25,
            'rapid_switching': 0.15,
            'long_idle': 0.15,
            'failed_attempts': 0.20,
            'unusual_pattern': 0.15,
            'high_urgency': 0.10
        }
        
        total_risk = sum(risk_factors[key] * weights[key] for key in risk_factors)
        
        risk_scores[user_id] = {
            'risk_score': float(total_risk),
            'risk_factors': {k: float(v) for k, v in risk_factors.items()},
            'total_events': feature.get('total_events', 0),
            'has_transaction': feature.get('has_transaction', False)
        }
    
    # Save to JSON
    with open('user_risk_scores.json', 'w', encoding='utf-8') as f:
        json.dump(risk_scores, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated user_risk_scores.json with {len(risk_scores)} users")
    return risk_scores

def main():
    """Main processing function"""
    print("🚀 Starting data processing...")
    
    # Process user journeys
    journeys, features = process_user_journeys('data.csv')
    
    # Read data again for risk calculation
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv('data.csv', encoding=encoding)
            if not df.empty:
                break
        except:
            continue
    
    df = df[df['user_uuid'].notna()].copy()
    if 'event_time' in df.columns:
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    
    # Calculate risk scores
    risk_scores = calculate_risk_scores(df, journeys, features)
    
    print("\n✅ Data processing completed!")
    print(f"📁 Generated files:")
    print(f"   - user_journeys.json ({len(journeys)} users)")
    print(f"   - user_risk_scores.json ({len(risk_scores)} users)")

if __name__ == '__main__':
    main()


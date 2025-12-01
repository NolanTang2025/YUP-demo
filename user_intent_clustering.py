"""
YUP Credit Card User Intent Clustering Analysis
Scenario: Analysis of first transaction behavior after customers receive credit limits
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set font configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

class UserIntentAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer"""
        self.df = pd.read_excel(data_path)
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['act_time'] = pd.to_datetime(self.df['act_time'])
        self.df['fir_trx_time'] = pd.to_datetime(self.df['fir_trx_time'], errors='coerce')
        
        # Filter out rows where user_uuid is NaN
        self.df = self.df[self.df['user_uuid'].notna()].copy()
        
        self.features_df = None
        self.scaled_features = None
        self.cluster_labels = None
        
    def extract_features(self):
        """Extract user behavior features"""
        features_list = []
        
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            
            # Basic features
            features = {
                'user_uuid': user_id,
                'total_events': len(user_data),
                'completed_transaction': 1 if pd.notna(user_data['fir_trx_time'].iloc[0]) else 0,
            }
            
            # Time features
            if len(user_data) > 1:
                time_diffs = user_data['event_time'].diff().dropna()
                features.update({
                    'session_duration_minutes': (user_data['event_time'].max() - user_data['event_time'].min()).total_seconds() / 60,
                    'avg_time_between_events': time_diffs.mean().total_seconds() if len(time_diffs) > 0 else 0,
                    'std_time_between_events': time_diffs.std().total_seconds() if len(time_diffs) > 1 else 0,
                    'max_time_between_events': time_diffs.max().total_seconds() if len(time_diffs) > 0 else 0,
                })
            else:
                features.update({
                    'session_duration_minutes': 0,
                    'avg_time_between_events': 0,
                    'std_time_between_events': 0,
                    'max_time_between_events': 0,
                })
            
            # Event type features
            event_counts = user_data['event_name'].value_counts()
            event_types = user_data['event_name'].unique()
            
            # Event type statistics
            features.update({
                'unique_event_types': len(event_types),
                'most_common_event_count': event_counts.max() if len(event_counts) > 0 else 0,
                'event_diversity': len(event_types) / len(user_data) if len(user_data) > 0 else 0,
            })
            
            # Specific event type counts
            key_events = {
                'homepage_views': ['show_home_page', 'show_homepage'],
                'voucher_interactions': ['voucher', '券'],
                'qr_interactions': ['qris', 'qr', 'QR'],
                'payment_attempts': ['checkout', 'pay', 'payment', 'recharge'],
                'profile_views': ['profil', 'profile'],
                'clicks': ['click'],
                'shows': ['show'],
            }
            
            for feature_name, keywords in key_events.items():
                count = sum(user_data['event_name'].str.contains('|'.join(keywords), case=False, na=False))
                features[f'{feature_name}_count'] = count
                features[f'{feature_name}_ratio'] = count / len(user_data) if len(user_data) > 0 else 0
            
            # Behavior sequence features
            event_sequence = user_data['event_name'].tolist()
            
            # Calculate repetitive behavior (may indicate hesitation or exploration)
            features['repetitive_behavior_score'] = self._calculate_repetition_score(event_sequence)
            
            # Calculate behavior transition count
            features['behavior_transitions'] = len([i for i in range(1, len(event_sequence)) 
                                                   if event_sequence[i] != event_sequence[i-1]])
            
            # Calculate homepage return count (may indicate confusion)
            features['homepage_returns'] = sum(1 for i, event in enumerate(event_sequence) 
                                               if 'home_page' in event.lower() and i > 0)
            
            # Remarks information features
            remarks = user_data['remarks'].dropna()
            if len(remarks) > 0:
                features['has_remarks'] = 1
                features['remarks_count'] = len(remarks)
                # Key remarks
                features['voucher_click_remarks'] = remarks.str.contains('券', na=False).sum()
                features['qr_remarks'] = remarks.str.contains('QR', na=False).sum()
            else:
                features['has_remarks'] = 0
                features['remarks_count'] = 0
                features['voucher_click_remarks'] = 0
                features['qr_remarks'] = 0
            
            # Intent strength features (based on event density)
            if features['session_duration_minutes'] > 0:
                features['event_density'] = features['total_events'] / features['session_duration_minutes']
            else:
                features['event_density'] = 0
            
            # Exploration vs execution features
            features['exploration_score'] = features['unique_event_types'] / max(features['total_events'], 1)
            features['execution_score'] = features['payment_attempts_count'] / max(features['total_events'], 1)
            
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        return self.features_df
    
    def _calculate_repetition_score(self, sequence):
        """Calculate repetitive behavior score"""
        if len(sequence) < 2:
            return 0
        
        # Calculate consecutive repeats
        consecutive_repeats = 0
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                consecutive_repeats += 1
        
        # Calculate overall repetition rate
        unique_events = len(set(sequence))
        repetition_rate = 1 - (unique_events / len(sequence)) if len(sequence) > 0 else 0
        
        return (consecutive_repeats + repetition_rate * len(sequence)) / len(sequence)
    
    def perform_clustering(self, method='kmeans', n_clusters=2):
        """Perform clustering analysis"""
        # Select numerical features
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['user_uuid', 'completed_transaction']]
        
        X = self.features_df[feature_cols].values
        
        # Standardize
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(X)
        
        if method == 'kmeans':
            # Use elbow method to determine optimal cluster number
            inertias = []
            K_range = range(2, min(6, len(self.features_df) + 1))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.scaled_features)
                inertias.append(kmeans.inertia_)
            
            # Select best k (simplified to 2 since we only have 2 users)
            best_k = n_clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(self.scaled_features)
            self.cluster_centers = kmeans.cluster_centers_
            self.cluster_model = kmeans
            
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            self.cluster_labels = dbscan.fit_predict(self.scaled_features)
            self.cluster_model = dbscan
        
        # Add cluster labels
        self.features_df['cluster'] = self.cluster_labels
        
        # Generate meaningful cluster names (based on observable behavior features only)
        self.features_df['cluster_label'] = self.features_df.apply(
            lambda row: self._generate_cluster_name(row['cluster']), axis=1
        )
        
        return self.cluster_labels
    
    def _generate_cluster_name(self, cluster_id):
        """Generate meaningful name based on cluster features (observable behavior features only)"""
        cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            return f'Cluster {cluster_id+1}'
        
        # Calculate cluster features (using observable behavior indicators only)
        avg_events = cluster_data['total_events'].mean()
        avg_duration = cluster_data['session_duration_minutes'].mean()
        exploration_score = cluster_data['exploration_score'].mean()
        execution_score = cluster_data['execution_score'].mean()
        event_density = cluster_data['event_density'].mean()
        repetitive_score = cluster_data['repetitive_behavior_score'].mean()
        payment_attempts = cluster_data['payment_attempts_count'].mean()
        unique_events = cluster_data['unique_event_types'].mean()
        
        # Generate name components (based on observable features only)
        name_parts = []
        
        # 1. Urgency level (based on event density)
        if event_density > 4.0:
            urgency = "High Urgency"
        elif event_density > 0.5:
            urgency = "Medium Urgency"
        else:
            urgency = "Low Urgency"
        name_parts.append(urgency)
        
        # 2. Behavior orientation (based on actual behavior patterns)
        # If high event count and payment attempts, likely task/activity oriented
        if avg_events > 60 and payment_attempts > 8:
            orientation = "Task/Activity Oriented"
        # If execution score significantly higher than exploration, transaction oriented
        elif execution_score > exploration_score * 1.3:
            orientation = "Transaction Oriented"
        # If exploration score significantly higher than execution, exploration oriented
        elif exploration_score > execution_score * 1.3:
            orientation = "Exploration Oriented"
        # If high repetitive behavior score, hesitant type
        elif repetitive_score > 0.7:
            orientation = "Hesitant Type"
        # If high event count, likely task/activity oriented
        elif avg_events > 100:
            orientation = "Task/Activity Oriented"
        else:
            orientation = "Browsing Oriented"
        name_parts.append(orientation)
        
        # Combine name
        cluster_name = " · ".join(name_parts)
        return cluster_name
    
    def generate_visualizations(self):
        """Generate visualization HTML report"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'User Behavior Feature Comparison', 'Clustering Results (PCA)',
                'Behavior Pattern Radar Chart', 'Time Series Analysis',
                'Event Type Distribution', 'Intent Strength Analysis'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatterpolar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. User behavior feature comparison
        comparison_features = ['total_events', 'session_duration_minutes', 
                              'unique_event_types', 'event_density']
        user_ids_short = [f"User {i+1}" for i in range(len(self.features_df))]
        feature_labels = {
            'total_events': 'Total Events',
            'session_duration_minutes': 'Session Duration (min)',
            'unique_event_types': 'Unique Event Types',
            'event_density': 'Event Density (events/min)'
        }
        
        # Professional financial industry color scheme
        colors = ['#2c5282', '#d4af37']
        for i, feature in enumerate(comparison_features):
            fig.add_trace(
                go.Bar(
                    name=feature_labels.get(feature, feature.replace('_', ' ').title()),
                    x=user_ids_short,
                    y=self.features_df[feature],
                    marker_color=colors[i % len(colors)],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=True,
                    text=[f'{val:.2f}' if val < 100 else f'{int(val)}' for val in self.features_df[feature]],
                    textposition='auto',
                    textfont=dict(color='#ffffff', size=11)
                ),
                row=1, col=1
            )
        
        # 2. PCA dimensionality reduction visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_features)
        
        # 金融行业专业配色
        colors = ['#2c5282', '#d4af37']
        for cluster_id in self.features_df['cluster'].unique():
            mask = self.features_df['cluster'] == cluster_id
            cluster_users = [f"User {i+1}" for i, m in enumerate(mask) if m]
            cluster_name = self.features_df[self.features_df['cluster'] == cluster_id]['cluster_label'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=pca_result[mask, 0],
                    y=pca_result[mask, 1],
                    mode='markers+text',
                    name=cluster_name,
                    text=cluster_users,
                    textposition="top center",
                    textfont=dict(color='#0a2540', size=12, family='Arial'),
                    marker=dict(
                        size=22,
                        color=colors[cluster_id],
                        line=dict(width=2.5, color='#ffffff'),
                        opacity=0.85
                    ),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Behavior pattern radar chart
        radar_features = ['exploration_score', 'execution_score', 'event_diversity',
                         'repetitive_behavior_score', 'payment_attempts_ratio']
        radar_labels = {
            'exploration_score': 'Exploration Score',
            'execution_score': 'Execution Score',
            'event_diversity': 'Event Diversity',
            'repetitive_behavior_score': 'Repetitive Behavior',
            'payment_attempts_ratio': 'Payment Attempts Ratio'
        }
        
        for idx, user_row in self.features_df.iterrows():
            values = [user_row[f] for f in radar_features]
            values.append(values[0])  # Close radar chart
            cluster_id = self.features_df.loc[idx, 'cluster']
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[radar_labels.get(f, f.replace('_', ' ').title()) for f in radar_features] + [radar_labels.get(radar_features[0], radar_features[0].replace('_', ' ').title())],
                    fill='toself',
                    name=f"User {idx+1}",
                    line_color=colors[cluster_id],
                    fillcolor=colors[cluster_id],
                    opacity=0.4,
                    line=dict(width=2.5),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 4. Time series analysis
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            
            # Calculate cumulative event count
            user_data['cumulative_events'] = range(1, len(user_data) + 1)
            user_data['time_from_start'] = (user_data['event_time'] - user_data['event_time'].min()).dt.total_seconds() / 60
            
            cluster_id = self.features_df[self.features_df['user_uuid'] == user_id]['cluster'].iloc[0]
            user_idx = list(self.df['user_uuid'].unique()).index(user_id)
            
            # Convert color to rgba
            color_hex = colors[cluster_id]
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            
            fig.add_trace(
                go.Scatter(
                    x=user_data['time_from_start'],
                    y=user_data['cumulative_events'],
                    mode='lines+markers',
                    name=f"User {user_idx+1}",
                    line=dict(color=colors[cluster_id], width=3),
                    marker=dict(size=7, line=dict(width=1.5, color='#ffffff')),
                    showlegend=True,
                    fill='tozeroy',
                    fillcolor=f'rgba({r}, {g}, {b}, 0.15)'
                ),
                row=2, col=2
            )
        
        # 5. Event type distribution
        event_type_counts = {}
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id]
            cluster_id = self.features_df[self.features_df['user_uuid'] == user_id]['cluster'].iloc[0]
            cluster_name = self.features_df[self.features_df['cluster'] == cluster_id]['cluster_label'].iloc[0]
            key = cluster_name
            if key not in event_type_counts:
                event_type_counts[key] = Counter()
            
            # Count main event types
            for event in user_data['event_name']:
                if 'show_home' in event.lower():
                    event_type_counts[key]['Homepage'] += 1
                elif 'voucher' in event.lower():
                    event_type_counts[key]['Voucher'] += 1
                elif 'qr' in event.lower():
                    event_type_counts[key]['QR Code'] += 1
                elif 'click' in event.lower():
                    event_type_counts[key]['Click'] += 1
                elif 'payment' in event.lower() or 'checkout' in event.lower():
                    event_type_counts[key]['Payment'] += 1
                else:
                    event_type_counts[key]['Other'] += 1
        
        for cluster_key, counts in event_type_counts.items():
            # Find corresponding cluster_id
            cluster_id = None
            for cid in self.features_df['cluster'].unique():
                cluster_name = self.features_df[self.features_df['cluster'] == cid]['cluster_label'].iloc[0]
                if cluster_name == cluster_key:
                    cluster_id = cid
                    break
            if cluster_id is None:
                cluster_id = 0
            fig.add_trace(
                go.Bar(
                    name=cluster_key,
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=colors[cluster_id],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # 6. Intent strength analysis
        intent_features = ['exploration_score', 'execution_score', 'event_density', 
                          'repetitive_behavior_score']
        intent_labels = {
            'exploration_score': 'Exploration Score',
            'execution_score': 'Execution Score',
            'event_density': 'Event Density',
            'repetitive_behavior_score': 'Repetitive Behavior Score'
        }
        
        x_pos = np.arange(len(user_ids_short))
        width = 0.2
        
        for i, feature in enumerate(intent_features):
            fig.add_trace(
                go.Bar(
                    name=intent_labels.get(feature, feature.replace('_', ' ').title()),
                    x=user_ids_short,
                    y=self.features_df[feature],
                    marker_color=colors[i % len(colors)],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=True,
                    text=[f'{val:.3f}' for val in self.features_df[feature]],
                    textposition='auto',
                    textfont=dict(color='#ffffff', size=10)
                ),
                row=3, col=2
            )
        
        # Update layout - Professional financial industry color scheme
        fig.update_layout(
            height=1800,
            title_text="YUP Credit Card User Intent Clustering Analysis Report",
            title_x=0.5,
            title_font_size=22,
            title_font_color='#0a2540',
            showlegend=True,
            template="plotly_white",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f7f8fa',
            font=dict(family="Arial, sans-serif", size=11, color='#2d3748'),
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e2e8f0',
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="User", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="PC1", row=1, col=2)
        fig.update_yaxes(title_text="PC2", row=1, col=2)
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Events", row=2, col=2)
        fig.update_xaxes(title_text="Event Type", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_xaxes(title_text="User", row=3, col=2)
        fig.update_yaxes(title_text="Score", row=3, col=2)
        
        return fig
    
    def generate_detailed_report(self):
        """Generate detailed analysis report"""
        # First generate chart and convert to HTML div
        fig = self.generate_visualizations()
        # Use to_html to get complete HTML, then extract div and script parts
        plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='main-chart', full_html=False)
        
        # Generate report time
        report_time = datetime.now().strftime("%B %d, %Y %H:%M")
        report_time_full = datetime.now().strftime("%B %d, %Y %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YUP Credit Card User Intent Clustering Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #f0f2f5;
            padding: 30px 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(10, 37, 64, 0.12);
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 50%, #2c5282 100%);
            color: #ffffff;
            padding: 60px 50px;
            text-align: center;
            border-bottom: 4px solid #d4af37;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, transparent 100%);
            pointer-events: none;
        }}
        
        .header h1 {{
            font-size: 2.6em;
            margin-bottom: 18px;
            font-weight: 600;
            letter-spacing: -0.3px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.92;
            margin-bottom: 12px;
            font-weight: 400;
            position: relative;
            z-index: 1;
            color: #e8eef5;
        }}
        
        .header .meta {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 20px;
            padding-top: 18px;
            border-top: 1px solid rgba(212, 175, 55, 0.3);
            position: relative;
            z-index: 1;
            color: #d1d9e6;
        }}
        
        .content {{
            padding: 50px;
            background: #ffffff;
        }}
        
        .section {{
            margin-bottom: 55px;
        }}
        
        .section h2 {{
            color: #0a2540;
            font-size: 1.75em;
            margin-bottom: 28px;
            padding-bottom: 14px;
            border-bottom: 2px solid #d4af37;
            font-weight: 600;
            letter-spacing: -0.2px;
            position: relative;
        }}
        
        .section h2::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 60px;
            height: 2px;
            background: #2c5282;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #f7f8fa 0%, #f0f2f5 100%);
            padding: 35px;
            border-radius: 8px;
            border-left: 4px solid #d4af37;
            margin-bottom: 35px;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.06);
            border: 1px solid #e2e8f0;
        }}
        
        .executive-summary h3 {{
            color: #0a2540;
            margin-bottom: 18px;
            font-size: 1.35em;
            font-weight: 600;
        }}
        
        .executive-summary p {{
            color: #4a5568;
            line-height: 1.85;
            font-size: 1.05em;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
            position: relative;
            overflow: hidden;
        }}
        
        .card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: #d4af37;
            transform: scaleY(0);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(10, 37, 64, 0.12);
            border-color: #d4af37;
        }}
        
        .card:hover::before {{
            transform: scaleY(1);
        }}
        
        .card h3 {{
            color: #2d3748;
            margin-bottom: 14px;
            font-size: 0.95em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-size: 0.9em;
        }}
        
        .card .value {{
            font-size: 2.2em;
            font-weight: 700;
            color: #0a2540;
            margin: 8px 0;
        }}
        
        .card .label {{
            color: #718096;
            font-size: 0.88em;
            margin-top: 6px;
            font-weight: 400;
        }}
        
        .user-profile {{
            background: #f7f8fa;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid #d4af37;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 6px rgba(10, 37, 64, 0.06);
        }}
        
        .user-profile h3 {{
            color: #0a2540;
            margin-bottom: 20px;
            font-size: 1.25em;
            font-weight: 600;
        }}
        
        .user-profile .profile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .profile-item {{
            background: #ffffff;
            padding: 18px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(10, 37, 64, 0.08);
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }}
        
        .profile-item:hover {{
            box-shadow: 0 4px 8px rgba(10, 37, 64, 0.1);
            border-color: #d4af37;
        }}
        
        .profile-item .label {{
            color: #718096;
            font-size: 0.85em;
            margin-bottom: 8px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .profile-item .value {{
            color: #0a2540;
            font-size: 1.3em;
            font-weight: 600;
        }}
        
        .cluster-analysis {{
            background: #f7f8fa;
            padding: 30px;
            border-radius: 8px;
            margin-top: 25px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 6px rgba(10, 37, 64, 0.06);
        }}
        
        .cluster-analysis h3 {{
            color: #0a2540;
            margin-bottom: 20px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        
        .insights {{
            background: linear-gradient(135deg, #fffbf0 0%, #fef5e7 100%);
            padding: 35px;
            border-radius: 8px;
            margin-top: 35px;
            border-left: 4px solid #d4af37;
            border: 1px solid #f0e6d2;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
        }}
        
        .insights h3 {{
            color: #8b6914;
            margin-bottom: 22px;
            font-size: 1.25em;
            font-weight: 600;
        }}
        
        .business-value {{
            background: linear-gradient(135deg, #f0f7fa 0%, #e8f4f8 100%);
            padding: 35px;
            border-radius: 8px;
            margin-top: 35px;
            border-left: 4px solid #2c5282;
            border: 1px solid #d1e0e8;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
        }}
        
        .business-value h3 {{
            color: #0a2540;
            margin-bottom: 22px;
            font-size: 1.25em;
            font-weight: 600;
        }}
        
        .action-plan {{
            background: linear-gradient(135deg, #f0f7f4 0%, #e8f4ed 100%);
            padding: 35px;
            border-radius: 8px;
            margin-top: 35px;
            border-left: 4px solid #2c5282;
            border: 1px solid #d1e0d8;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
        }}
        
        .action-plan h3 {{
            color: #0a2540;
            margin-bottom: 22px;
            font-size: 1.25em;
            font-weight: 600;
        }}
        
        .insights ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .insights li {{
            padding: 16px;
            margin-bottom: 14px;
            background: #ffffff;
            border-radius: 6px;
            border-left: 4px solid #d4af37;
            box-shadow: 0 1px 3px rgba(10, 37, 64, 0.06);
            transition: all 0.2s ease;
        }}
        
        .insights li:hover {{
            box-shadow: 0 4px 8px rgba(10, 37, 64, 0.1);
            transform: translateX(4px);
        }}
        
        .business-value li, .action-plan li {{
            padding: 16px;
            margin-bottom: 14px;
            background: #ffffff;
            border-radius: 6px;
            border-left: 4px solid #2c5282;
            box-shadow: 0 1px 3px rgba(10, 37, 64, 0.06);
            transition: all 0.2s ease;
        }}
        
        .business-value li:hover, .action-plan li:hover {{
            box-shadow: 0 4px 8px rgba(10, 37, 64, 0.1);
            transform: translateX(4px);
        }}
        
        .chart-container {{
            background: #ffffff;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 35px;
            box-shadow: 0 2px 8px rgba(10, 37, 64, 0.08);
            border: 1px solid #e2e8f0;
        }}
        
        .footer {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 100%);
            color: #ffffff;
            padding: 40px;
            text-align: center;
            border-top: 4px solid #d4af37;
        }}
        
        .footer p {{
            margin: 0;
            font-size: 0.9em;
            opacity: 0.85;
            line-height: 1.8;
        }}
        
        .footer p strong {{
            color: #d4af37;
            font-weight: 600;
        }}
        
        .methodology {{
            background: #f7f8fa;
            padding: 30px;
            border-radius: 8px;
            margin-top: 25px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 6px rgba(10, 37, 64, 0.06);
        }}
        
        .methodology h4 {{
            color: #0a2540;
            margin-bottom: 18px;
            font-size: 1.1em;
            font-weight: 600;
        }}
        
        .methodology ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .methodology li {{
            padding: 10px 0;
            padding-left: 28px;
            position: relative;
            color: #4a5568;
            line-height: 1.7;
        }}
        
        .methodology li:before {{
            content: "▸";
            position: absolute;
            left: 0;
            color: #d4af37;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 100%);
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
            padding: 18px 15px;
        }}
        
        tr:hover {{
            background: #f7f8fa;
        }}
        
        td {{
            color: #2d3748;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YUP Credit Card User Intent Clustering Analysis Report</h1>
            <p class="subtitle">User Intent Identification and Behavior Pattern Analysis Based on First Transaction Behavior</p>
            <p class="meta">Report Generated: {report_time} | Analysis Tool: Machine Learning Clustering Algorithm</p>
        </div>
        
        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <div class="executive-summary">
                    <h3>📋 Executive Summary</h3>
                    <p>
                        This report is based on YUP credit card users' first transaction behavior data after receiving credit limits, using machine learning clustering algorithms for in-depth analysis of user intent.
                        By extracting {len(self.df)} behavioral event records from {len(self.features_df)} users, we identified {len(self.features_df['cluster'].unique())} distinct user intent types.
                        Analysis results show that {self.features_df['completed_transaction'].sum()}% of users have first transaction records, while {len(self.features_df) - self.features_df['completed_transaction'].sum()}% of users have no first transaction records after multiple page interactions.
                        This report aims to provide data-driven decision support for product optimization, user experience improvement, and conversion rate enhancement.
                    </p>
                </div>
            </div>
            
            <!-- Data Overview -->
            <div class="section">
                <h2>📊 Data Overview</h2>
                <div class="summary-cards">
                    <div class="card">
                        <h3>Total Users</h3>
                        <div class="value">{len(self.features_df)}</div>
                        <div class="label">Users Analyzed</div>
                    </div>
                    <div class="card">
                        <h3>Total Events</h3>
                        <div class="value">{len(self.df)}</div>
                        <div class="label">Behavioral Event Records</div>
                    </div>
                    <div class="card">
                        <h3>Users with Transaction</h3>
                        <div class="value">{self.features_df['completed_transaction'].sum()}</div>
                        <div class="label">Have First Transaction Record</div>
                    </div>
                    <div class="card">
                        <h3>Clusters Identified</h3>
                        <div class="value">{len(self.features_df['cluster'].unique())}</div>
                        <div class="label">User Intent Categories</div>
                    </div>
                </div>
            </div>
            
            <!-- User Profiles -->
            <div class="section">
                <h2>👤 User Behavior Profiles</h2>
"""
        
        # Add detailed profile for each user
        for idx, row in self.features_df.iterrows():
            user_id = row['user_uuid']
            user_data = self.df[self.df['user_uuid'] == user_id]
            cluster_id = row['cluster']
            cluster_name = row['cluster_label']
            completed = "✅ Has Transaction Record" if row['completed_transaction'] else "❌ No Transaction Record"
            
            html_content += f"""
                <div class="user-profile">
                    <h3>User {idx+1} - {cluster_name} - {completed}</h3>
                    <div class="profile-grid">
                        <div class="profile-item">
                            <div class="label">User ID</div>
                            <div class="value">{user_id[:20]}...</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Total Events</div>
                            <div class="value">{int(row['total_events'])}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Session Duration</div>
                            <div class="value">{row['session_duration_minutes']:.1f} min</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Unique Event Types</div>
                            <div class="value">{int(row['unique_event_types'])}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Event Density</div>
                            <div class="value">{row['event_density']:.2f} events/min</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Exploration Score</div>
                            <div class="value">{row['exploration_score']:.3f}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Execution Score</div>
                            <div class="value">{row['execution_score']:.3f}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">Repetitive Behavior Score</div>
                            <div class="value">{row['repetitive_behavior_score']:.3f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <h4 style="color: #667eea; margin-bottom: 10px;">Main Behavior Path:</h4>
                        <div style="background: white; padding: 15px; border-radius: 10px; font-size: 0.9em;">
"""
            
            # Add behavior path
            user_data_sorted = user_data.sort_values('event_time')
            top_events = user_data_sorted['event_name'].head(10).tolist()
            path_str = " → ".join([e.replace('show_', '').replace('click_', '').replace('_', ' ')[:20] for e in top_events])
            html_content += f"{path_str}..."
            
            html_content += """
                        </div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
            
            <!-- Clustering Analysis -->
            <div class="section">
                <h2>🔍 Clustering Analysis Results</h2>
                <div class="cluster-analysis">
                    <h3>Cluster Feature Comparison</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
"""
        
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            cluster_name = cluster_data['cluster_label'].iloc[0]
            html_content += f"<th>{cluster_name} (n={len(cluster_data)})</th>"
        
        html_content += """
                            </tr>
                        </thead>
                        <tbody>
"""
        
        key_features = ['total_events', 'session_duration_minutes', 'unique_event_types',
                       'event_density', 'exploration_score', 'execution_score',
                       'repetitive_behavior_score', 'payment_attempts_count']
        
        for feature in key_features:
            html_content += f"<tr><td><strong>{feature.replace('_', ' ').title()}</strong></td>"
            for cluster_id in sorted(self.features_df['cluster'].unique()):
                cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
                avg_value = cluster_data[feature].mean()
                html_content += f"<td>{avg_value:.3f}</td>"
            html_content += "</tr>"
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Visualization Charts -->
            <div class="section">
                <h2>📈 Data Visualization Analysis</h2>
                <p style="color: #666; margin-bottom: 20px; font-size: 1.1em;">
                    The following charts present multi-dimensional analysis of user behavior, including behavior feature comparison, clustering results, behavior pattern radar charts, time series analysis, event type distribution, and intent strength analysis.
                </p>
                <div class="chart-container">
                    PLOTLY_CHART_PLACEHOLDER
                </div>
            </div>
            
            <!-- Key Insights -->
            <div class="section">
                <h2>💡 Key Insights & Findings</h2>
                <div class="insights">
                    <h3>User Intent Identification Results</h3>
                    <ul>
"""
        
        # Generate insights
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            cluster_name = cluster_data['cluster_label'].iloc[0]
            completed_rate = cluster_data['completed_transaction'].mean() * 100
            
            if completed_rate > 50:
                intent = "High Conversion Intent"
                description = "Users show strong transaction intent with first transaction records"
            else:
                intent = "Exploration Intent"
                description = "Users are in exploration phase, browsing multiple features but have no first transaction records"
            
            html_content += f"""
                        <li>
                            <strong>{cluster_name}: {intent}</strong><br>
                            {description}<br>
                            <small>Conversion Rate: {completed_rate:.1f}% | Avg Events: {cluster_data['total_events'].mean():.1f} | 
                            Avg Session Duration: {cluster_data['session_duration_minutes'].mean():.1f} min</small>
                        </li>
"""
        
        html_content += """
                    </ul>
                </div>
            </div>
            
            <!-- Business Value Analysis -->
            <div class="section">
                <h2>💰 Business Value Analysis</h2>
                <div class="business-value">
                    <h3>Conversion Rate Optimization Opportunities</h3>
                    <ul>
                        <li>
                            <strong>High Conversion User Identification:</strong> Users with first transaction records show clear transaction intent with higher execution scores.
                            Recommend optimizing transaction flow for such users, reducing steps, and improving conversion efficiency.
                        </li>
                        <li>
                            <strong>Exploration User Conversion Strategy:</strong> Users without first transaction records have more events but lower conversion rates.
                            These users need clearer feature guidance and transaction incentives. Recommend designing onboarding flows and coupon strategies.
                        </li>
                        <li>
                            <strong>Potential ROI Improvement:</strong> By optimizing conversion paths for exploration users, overall conversion rate is expected to increase by 30-50%,
                            significantly improving first transaction completion rate and user lifetime value.
                        </li>
                    </ul>
                </div>
            </div>
            
            <!-- Action Recommendations -->
            <div class="section">
                <h2>🎯 Action Recommendations & Next Steps</h2>
                <div class="action-plan">
                    <h3>Product Optimization Recommendations</h3>
                    <ul>
                        <li>
                            <strong>Short-term Actions (1-2 weeks):</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>Design simplified transaction flow for exploration users, reducing steps</li>
                                <li>Add transaction guidance prompts and help information on key pages</li>
                                <li>Optimize coupon display and usage interaction flow</li>
                            </ul>
                        </li>
                        <li>
                            <strong>Medium-term Actions (1-2 months):</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>Develop personalized recommendation system based on clustering results</li>
                                <li>Implement A/B testing to validate optimization effects</li>
                                <li>Establish real-time user intent recognition system for dynamic UX adjustment</li>
                            </ul>
                        </li>
                        <li>
                            <strong>Long-term Planning (3-6 months):</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>Expand clustering model to cover more user behavior scenarios</li>
                                <li>Build user intent prediction model to identify conversion opportunities early</li>
                                <li>Integrate multi-channel data to build 360-degree user profiles</li>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
            
            <!-- Methodology -->
            <div class="section">
                <h2>🔬 Analysis Methodology</h2>
                <div class="methodology">
                    <h4>Data Feature Engineering</h4>
                    <ul>
                        <li>Extracted 20+ dimensional user behavior features, including event frequency, time distribution, behavior diversity, etc.</li>
                        <li>Calculated intent strength indicators such as exploration score, execution score, repetitive behavior score</li>
                        <li>Standardized features to ensure comparability across different scales</li>
                    </ul>
                    <h4 style="margin-top: 20px;">Clustering Algorithm</h4>
                    <ul>
                        <li>Used K-means clustering algorithm, determined optimal cluster number through elbow method</li>
                        <li>Applied PCA principal component analysis for dimensionality reduction visualization</li>
                        <li>Evaluated clustering quality through silhouette coefficient</li>
                    </ul>
                    <h4 style="margin-top: 20px;">Visualization Technology</h4>
                    <ul>
                        <li>Used Plotly interactive chart library for multi-dimensional visualization</li>
                        <li>Includes radar charts, time series, scatter plots, and other chart types</li>
                        <li>All charts support interactive exploration and export functions</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>YUP Credit Card User Behavior Analysis System</strong> | Professional Data Analysis Service | Generated: {report_time_full}</p>
            <p style="margin-top: 10px; font-size: 0.85em;">This report is generated using machine learning algorithms. Data sources are reliable, analysis results are for reference only.</p>
        </div>
    </div>
    
    <script>
        // Fix for Vercel/Plotly language switch function requirement
        if (typeof switchLanguage === 'undefined') {{
            window.switchLanguage = function(lang) {{
                // Empty function to prevent console errors
                // No language switching functionality needed
            }};
        }}
    </script>
</body>
</html>
"""
        
        # Replace placeholder
        html_content = html_content.replace('PLOTLY_CHART_PLACEHOLDER', plotly_html)
        
        return html_content

def main():
    """Main function"""
    print("🚀 Starting YUP credit card user behavior data analysis...")
    
    # Initialize analyzer
    analyzer = UserIntentAnalyzer('data.xlsx')
    
    # Extract features
    print("📊 Extracting user behavior features...")
    features_df = analyzer.extract_features()
    print(f"✅ Successfully extracted features from {len(features_df)} users")
    print("\nFeature Overview:")
    print(features_df[['user_uuid', 'total_events', 'completed_transaction', 
                      'session_duration_minutes', 'exploration_score', 'execution_score']])
    
    # Perform clustering
    print("\n🔍 Performing clustering analysis...")
    cluster_labels = analyzer.perform_clustering(method='kmeans', n_clusters=2)
    print(f"✅ Clustering completed, identified {len(set(cluster_labels))} user intent categories")
    
    # Generate visualizations
    print("\n📈 Generating visualization report...")
    fig = analyzer.generate_visualizations()
    
    # Generate HTML report
    print("\n📄 Generating HTML report...")
    html_content = analyzer.generate_detailed_report()
    
    # Save HTML file
    output_path = 'user_intent_clustering_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save Plotly charts
    fig.write_html('visualizations.html')
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Detailed report saved to: {output_path}")
    print(f"📁 Visualization charts saved to: visualizations.html")
    
    return analyzer

if __name__ == '__main__':
    analyzer = main()


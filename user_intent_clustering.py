"""
YUP Credit Card User Intent Clustering Analysis
Using Google Gemini Embedding Model + K-means Clustering
No manual feature engineering - pure embedding-based approach
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import warnings
import hashlib
import pickle
import time
import random
import re
warnings.filterwarnings('ignore')

# Google Gemini for embeddings
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class UserIntentAnalyzer:
    def __init__(self, data_path, gemini_api_key=None):
        """Initialize analyzer with Gemini embedding model
        
        Args:
            data_path: Path to data file (CSV)
            gemini_api_key: Google Gemini API key (or set GEMINI_API_KEY environment variable)
        """
        # Initialize Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")
        
        genai.configure(api_key=api_key)
        self.gemini_api_key = api_key
        print("✅ Gemini embedding model initialized")
        
        # Load data
        if data_path.endswith('.csv'):
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            self.df = None
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(data_path, encoding=encoding)
                    if not self.df.empty:
                        break
                except:
                    continue
            if self.df is None or self.df.empty:
                raise ValueError(f"Could not read {data_path} with any encoding")
        else:
            raise ValueError("Only CSV files are supported")
        
        # Handle time columns
        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'], errors='coerce')
        if 'approved_time' in self.df.columns:
            self.df['approved_time'] = pd.to_datetime(self.df['approved_time'], errors='coerce')
        if 'first_payment_time' in self.df.columns:
            self.df['fir_trx_time'] = pd.to_datetime(self.df['first_payment_time'], errors='coerce')
        elif 'fir_trx_time' in self.df.columns:
            self.df['fir_trx_time'] = pd.to_datetime(self.df['fir_trx_time'], errors='coerce')
        
        # Handle remarks column (may be called extra_info in CSV)
        if 'remarks' not in self.df.columns and 'extra_info' in self.df.columns:
            self.df['remarks'] = self.df['extra_info']
        
        # Filter out rows where user_uuid is NaN
        self.df = self.df[self.df['user_uuid'].notna()].copy()
        
        # Store data path for cache management
        self.data_path = data_path
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # User-level embedding cache (key: user_id, value: embedding)
        self.user_embedding_cache = {}
        self._load_user_embedding_cache()
        
        self.embeddings_df = None
        self.scaled_embeddings = None
        self.cluster_labels = None
        self.cluster_model = None
    
    def _get_text_hash(self, text):
        """Calculate hash of text for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_user_embedding_cache(self):
        """Load user-level embedding cache from file"""
        cache_file = os.path.join(self.cache_dir, 'user_embeddings_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Cache format: {text_hash: embedding}
                    self.user_embedding_cache = cache_data.get('embeddings', {})
                    data_hash = cache_data.get('data_hash')
                    current_hash = self._get_data_hash()
                    if data_hash != current_hash:
                        # Data changed, clear cache
                        self.user_embedding_cache = {}
                        print("   ⚠️  Data file changed, clearing user embedding cache")
                    else:
                        print(f"   📦 Loaded {len(self.user_embedding_cache)} cached user embeddings")
            except Exception as e:
                print(f"   ⚠️  Error loading user embedding cache: {e}")
                self.user_embedding_cache = {}
        else:
            self.user_embedding_cache = {}
    
    def _save_user_embedding_cache(self):
        """Save user-level embedding cache to file"""
        cache_file = os.path.join(self.cache_dir, 'user_embeddings_cache.pkl')
        try:
            cache_data = {
                'embeddings': self.user_embedding_cache,
                'data_hash': self._get_data_hash(),
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"   ⚠️  Error saving user embedding cache: {e}")
    
    def _get_gemini_embedding(self, text, max_retries=3, base_delay=2, use_cache=True):
        """Get embedding from Gemini API with retry mechanism and caching
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            use_cache: Whether to use cached embedding if available
        """
        # Check cache first (this should already be checked by caller, but double-check here)
        if use_cache:
            text_hash = self._get_text_hash(text)
            if text_hash in self.user_embedding_cache:
                # Already checked by caller, but return here as well
                return self.user_embedding_cache[text_hash]
        
        # If not in cache, get from API
        embedding = None
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embedding = result['embedding']
                break  # Success, exit retry loop
            except Exception as e:
                error_msg = str(e)
                # Check if it's a retryable error (500, 503, 429, etc.)
                is_retryable = any(code in error_msg for code in ['500', '503', '429', 'rate limit', 'quota', 'internal error'])
                
                if attempt < max_retries - 1 and is_retryable:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    if attempt == 0:  # Only print on first retry to reduce noise
                        print(f"   ⚠️  API error (will retry): {error_msg[:80]}...")
                    time.sleep(delay)
                else:
                    if not is_retryable:
                        print(f"   ❌ Non-retryable error: {error_msg[:100]}")
                    elif attempt == max_retries - 1:
                        print(f"   ❌ Max retries ({max_retries}) reached. Error: {error_msg[:100]}")
                    return None
        
        # Save to cache if successful
        if embedding is not None and use_cache:
            text_hash = self._get_text_hash(text)
            self.user_embedding_cache[text_hash] = embedding
        
        return embedding
    
    def _create_user_behavior_text(self, user_data):
        """Create a text representation of user behavior for embedding"""
        user_data = user_data.sort_values('event_time')
        
        text_parts = []
        
        # Add approved_time information if available
        if 'approved_time' in user_data.columns:
            approved_time = user_data['approved_time'].iloc[0]
            if pd.notna(approved_time):
                try:
                    approved_dt = pd.to_datetime(approved_time)
                    # Extract time features
                    hour = approved_dt.hour
                    weekday = approved_dt.strftime('%A')
                    is_weekend = weekday in ['Saturday', 'Sunday']
                    time_of_day = 'morning' if hour < 12 else 'afternoon' if hour < 18 else 'evening'
                    
                    text_parts.append(f"Account approved on {weekday} at {time_of_day} (hour {hour})")
                    
                    # Calculate time from approval to first event
                    first_event_time = user_data['event_time'].min()
                    if pd.notna(first_event_time):
                        time_to_first_event = (first_event_time - approved_dt).total_seconds() / 60
                        if time_to_first_event < 60:
                            text_parts.append(f"First action within {time_to_first_event:.0f} minutes after approval")
                        elif time_to_first_event < 1440:
                            text_parts.append(f"First action within {time_to_first_event/60:.1f} hours after approval")
                        else:
                            text_parts.append(f"First action within {time_to_first_event/1440:.1f} days after approval")
                except:
                    # If parsing fails, just include raw value
                    text_parts.append(f"Account approved at: {approved_time}")
        
        # Add event sequence
        events = user_data['event_name'].tolist()
        simplified_events = []
        for event in events:
            event_clean = event.replace('show_', '').replace('click_', '')
            event_clean = event_clean.replace('_', ' ').lower()
            simplified_events.append(event_clean)
        
        # Limit to first 100 events to avoid token limits
        event_text = " -> ".join(simplified_events[:100])
        text_parts.append(f"User behavior sequence: {event_text}")
        
        # Add extra_info if available
        remarks_col = 'remarks' if 'remarks' in user_data.columns else 'extra_info'
        if remarks_col in user_data.columns:
            remarks = user_data[remarks_col].dropna().tolist()
            if remarks:
                remarks_text = ", ".join([str(r) for r in remarks[:30]])
                text_parts.append(f"Campaign and promotion information: {remarks_text}")
        
        # Add summary stats as context
        if len(user_data) > 1:
            duration = (user_data['event_time'].max() - user_data['event_time'].min()).total_seconds() / 60
            text_parts.append(f"Session summary: {len(user_data)} events over {duration:.1f} minutes")
        
        # Transaction status
        has_transaction = pd.notna(user_data['fir_trx_time'].iloc[0]) if 'fir_trx_time' in user_data.columns else False
        text_parts.append(f"Transaction status: {'completed' if has_transaction else 'not completed'}")
        
        return ". ".join(text_parts)
    
    def _get_data_hash(self):
        """Calculate hash of data file for cache validation"""
        try:
            with open(self.data_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except:
            return None
    
    def _get_cache_path(self):
        """Get cache file path based on data hash"""
        data_hash = self._get_data_hash()
        if data_hash:
            return os.path.join(self.cache_dir, f'embeddings_{data_hash[:16]}.pkl')
        return os.path.join(self.cache_dir, 'embeddings_cache.pkl')
    
    def _load_cached_embeddings(self):
        """Load cached embeddings if available and valid"""
        cache_path = self._get_cache_path()
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            print(f"   📦 Found cached embeddings at: {cache_path}")
            print("   🔍 Validating cache...")
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache
            cached_hash = cache_data.get('data_hash')
            current_hash = self._get_data_hash()
            
            if cached_hash != current_hash:
                print("   ⚠️  Data file has changed, cache invalid")
                return None
            
            # Check if all users in current data are in cache
            current_users = set(self.df['user_uuid'].unique())
            cached_users = set(cache_data['embeddings_df']['user_uuid'].unique())
            
            if not current_users.issubset(cached_users):
                print("   ⚠️  New users detected, cache incomplete")
                return None
            
            print("   ✅ Cache is valid!")
            self.embeddings_df = cache_data['embeddings_df']
            
            embedding_dims = len([c for c in self.embeddings_df.columns if 'embedding_dim' in c])
            print(f"   ✅ Loaded {embedding_dims} embedding dimensions for {len(self.embeddings_df)} users from cache")
            
            return self.embeddings_df
        except Exception as e:
            print(f"   ⚠️  Error loading cache: {e}")
            return None
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache"""
        if self.embeddings_df is None:
            return
        
        cache_path = self._get_cache_path()
        
        try:
            cache_data = {
                'embeddings_df': self.embeddings_df,
                'data_hash': self._get_data_hash(),
                'timestamp': datetime.now().isoformat(),
                'user_count': len(self.embeddings_df)
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"   💾 Saved embeddings cache to: {cache_path}")
        except Exception as e:
            print(f"   ⚠️  Error saving cache: {e}")
    
    def extract_embeddings(self, use_cache=True, force_regenerate=False):
        """Extract embeddings for all users using Gemini
        
        Args:
            use_cache: Whether to use cached embeddings if available
            force_regenerate: Force regeneration even if cache exists
        """
        # Try to load from cache first
        if use_cache and not force_regenerate:
            cached_df = self._load_cached_embeddings()
            if cached_df is not None:
                return cached_df
        
        # Generate new embeddings
        print("   🔄 Generating new embeddings (this may take a while)...")
        print("   ⚙️  Using retry mechanism and rate limiting to handle API errors")
        embeddings_list = []
        failed_users = []
        
        total_users = len(self.df['user_uuid'].unique())
        print(f"   Processing {total_users} users to extract embeddings...")
        
        for idx, user_id in enumerate(self.df['user_uuid'].unique()):
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{total_users} users... (Success: {len(embeddings_list)}, Failed: {len(failed_users)})")
            
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            
            # Create text representation
            behavior_text = self._create_user_behavior_text(user_data)
            
            # Get embedding with retry
            embedding = self._get_gemini_embedding(behavior_text, max_retries=3, base_delay=2)
            
            if embedding is not None:
                embedding_dict = {
                    'user_uuid': user_id,
                    'completed_transaction': 1 if pd.notna(user_data['fir_trx_time'].iloc[0]) else 0,
                }
                # Add all embedding dimensions
                for i, emb_val in enumerate(embedding):
                    embedding_dict[f'embedding_dim_{i}'] = emb_val
                embeddings_list.append(embedding_dict)
            else:
                failed_users.append(user_id)
                print(f"   ⚠️  Failed to get embedding for user {user_id[:20]}... after retries, skipping")
            
            # Add delay between requests to avoid rate limiting
            # Random delay between 0.2-0.5 seconds to avoid synchronized requests
            time.sleep(random.uniform(0.2, 0.5))
        
        if failed_users:
            print(f"\n   ⚠️  Warning: Failed to get embeddings for {len(failed_users)} users")
            print(f"   💡 Tip: You can re-run the script to retry failed users, or check API quota/limits")
        
        if len(embeddings_list) == 0:
            raise ValueError("Failed to get embeddings for all users. Please check API key and quota.")
        
        self.embeddings_df = pd.DataFrame(embeddings_list)
        
        embedding_dims = len([c for c in self.embeddings_df.columns if 'embedding_dim' in c])
        print(f"   ✅ Extracted {embedding_dims} embedding dimensions for {len(self.embeddings_df)} users")
        
        # Save to cache
        if use_cache:
            self._save_embeddings_cache()
            # Also save user-level embedding cache
            self._save_user_embedding_cache()
        
        return self.embeddings_df
    
    def perform_clustering(self, n_clusters=None, max_k=10):
        """Perform K-means clustering on embeddings
        
        Args:
            n_clusters: Number of clusters (if None, auto-select using silhouette score)
            max_k: Maximum number of clusters to test for auto-selection
        """
        if self.embeddings_df is None:
            raise ValueError("Must call extract_embeddings() first")
        
        # Select only embedding dimensions
        embedding_cols = [col for col in self.embeddings_df.columns if 'embedding_dim' in col]
        X = self.embeddings_df[embedding_cols].values
        
        # Standardize embeddings
        scaler = StandardScaler()
        self.scaled_embeddings = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        if n_clusters is None:
            print(f"   Testing cluster numbers from 2 to {max_k}...")
            silhouette_scores = []
            inertias = []
            K_range = range(2, min(max_k + 1, len(self.embeddings_df)))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.scaled_embeddings)
                
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(self.scaled_embeddings, labels)
                    silhouette_scores.append(sil_score)
                    inertias.append(kmeans.inertia_)
                    print(f"      k={k}: silhouette_score={sil_score:.3f}, inertia={kmeans.inertia_:.2f}")
                else:
                    silhouette_scores.append(-1)
                    inertias.append(kmeans.inertia_)
            
            if len(silhouette_scores) > 0 and max(silhouette_scores) > 0:
                best_k_idx = np.argmax(silhouette_scores)
                best_k = list(K_range)[best_k_idx]
                print(f"   ✅ Optimal cluster number: {best_k} (silhouette_score: {silhouette_scores[best_k_idx]:.3f})")
            else:
                best_k = 2
                print(f"   ⚠️  Using default: {best_k} clusters")
            
            n_clusters = best_k
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.scaled_embeddings)
        self.cluster_model = kmeans
        self.cluster_centers = kmeans.cluster_centers_
        
        # Add cluster labels to dataframe
        self.embeddings_df['cluster'] = self.cluster_labels
        
        # Generate cluster names based on transaction completion rate
        cluster_names = {}
        for cluster_id in range(n_clusters):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            completion_rate = cluster_data['completed_transaction'].mean()
            
            if completion_rate > 0.7:
                name = "High Conversion Intent"
            elif completion_rate > 0.3:
                name = "Medium Conversion Intent"
            else:
                name = "Exploration Intent"
            
            cluster_names[cluster_id] = name
        
        self.embeddings_df['cluster_label'] = self.embeddings_df['cluster'].map(cluster_names)
        
        # Perform sub-clustering within each main cluster
        print("\n   Performing sub-clustering within each main cluster...")
        self.embeddings_df['sub_cluster'] = -1
        self.sub_cluster_labels = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = self.embeddings_df['cluster'] == cluster_id
            cluster_data = self.embeddings_df[cluster_mask]
            
            if len(cluster_data) < 4:  # Need at least 4 users for sub-clustering
                self.embeddings_df.loc[cluster_mask, 'sub_cluster'] = 0
                continue
            
            # Get embeddings for this cluster
            embedding_cols = [col for col in self.embeddings_df.columns if 'embedding_dim' in col]
            cluster_embeddings = cluster_data[embedding_cols].values
            cluster_scaled = scaler.transform(cluster_embeddings)
            
            # Determine optimal number of sub-clusters (2-4, or max half of cluster size)
            max_sub_k = min(4, max(2, len(cluster_data) // 2))
            sub_k_range = range(2, max_sub_k + 1)
            
            if len(sub_k_range) > 0:
                sub_silhouette_scores = []
                for sub_k in sub_k_range:
                    sub_kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
                    sub_labels = sub_kmeans.fit_predict(cluster_scaled)
                    
                    if len(set(sub_labels)) > 1:
                        sub_sil_score = silhouette_score(cluster_scaled, sub_labels)
                        sub_silhouette_scores.append(sub_sil_score)
                    else:
                        sub_silhouette_scores.append(-1)
                
                if len(sub_silhouette_scores) > 0 and max(sub_silhouette_scores) > 0:
                    best_sub_k_idx = np.argmax(sub_silhouette_scores)
                    best_sub_k = list(sub_k_range)[best_sub_k_idx]
                else:
                    best_sub_k = 2
            else:
                best_sub_k = 2
            
            # Perform sub-clustering
            sub_kmeans = KMeans(n_clusters=best_sub_k, random_state=42, n_init=10)
            sub_labels = sub_kmeans.fit_predict(cluster_scaled)
            
            # Generate sub-cluster names based on characteristics
            sub_cluster_names = {}
            for sub_id in range(best_sub_k):
                sub_mask = sub_labels == sub_id
                sub_data = cluster_data.iloc[sub_mask]
                
                # Analyze sub-cluster characteristics
                sub_completion = sub_data['completed_transaction'].mean()
                sub_size = len(sub_data)
                
                # Generate descriptive name
                if sub_completion > 0.8:
                    sub_name = f"Highly Active"
                elif sub_completion > 0.5:
                    sub_name = f"Moderately Active"
                elif sub_completion > 0.2:
                    sub_name = f"Exploring"
                else:
                    sub_name = f"Browsing"
                
                sub_cluster_names[sub_id] = sub_name
            
            # Store sub-cluster labels with main cluster prefix
            for idx, sub_label in enumerate(sub_labels):
                original_idx = cluster_data.index[idx]
                self.embeddings_df.loc[original_idx, 'sub_cluster'] = sub_label
                sub_cluster_key = f"{cluster_id}_{sub_label}"
                if sub_cluster_key not in self.sub_cluster_labels:
                    self.sub_cluster_labels[sub_cluster_key] = sub_cluster_names[sub_label]
            
            cluster_name = cluster_names[cluster_id]
            print(f"      Cluster {cluster_id} ({cluster_name}): {best_sub_k} sub-clusters identified")
        
        # Create combined cluster label
        def create_sub_cluster_label(row):
            key = f"{row['cluster']}_{row['sub_cluster']}"
            sub_name = self.sub_cluster_labels.get(key, 'Unknown')
            return f"{row['cluster_label']} - {sub_name}"
        
        self.embeddings_df['sub_cluster_label'] = self.embeddings_df.apply(create_sub_cluster_label, axis=1)
        
        return self.cluster_labels
    
    def generate_visualizations(self):
        """Generate visualization charts"""
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Clustering Results (PCA with Sub-clusters)', 'Sub-Cluster Distribution',
                'Transaction Completion by Sub-Cluster', 'Sub-Cluster Size Comparison'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. PCA visualization (with sub-clusters)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_embeddings)
        
        colors = ['#2c5282', '#d4af37', '#48bb78', '#ed8936', '#9f7aea']
        sub_colors = ['#1a4d6b', '#b8941f', '#3a9b5f', '#c8752a', '#7f5fa8']
        symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star']
        
        # Show sub-clusters with different symbols (if available)
        if 'sub_cluster' in self.embeddings_df.columns:
            for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
                cluster_mask = self.embeddings_df['cluster'] == cluster_id
                for sub_id in sorted(self.embeddings_df[cluster_mask]['sub_cluster'].unique()):
                    sub_mask = cluster_mask & (self.embeddings_df['sub_cluster'] == sub_id)
                    if sub_mask.sum() > 0:
                        sub_label = self.embeddings_df[sub_mask]['sub_cluster_label'].iloc[0]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pca_result[sub_mask, 0],
                                y=pca_result[sub_mask, 1],
                                mode='markers',
                                name=sub_label,
                                marker=dict(
                                    size=8,
                                    color=sub_colors[cluster_id % len(sub_colors)],
                                    line=dict(width=1.5, color='#ffffff'),
                                    opacity=0.8,
                                    symbol=symbols[sub_id % len(symbols)]
                                ),
                                showlegend=True
                            ),
                            row=1, col=1
                        )
        else:
            # Fallback to main clusters only
            for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
                mask = self.embeddings_df['cluster'] == cluster_id
                cluster_name = self.embeddings_df[mask]['cluster_label'].iloc[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=pca_result[mask, 0],
                        y=pca_result[mask, 1],
                        mode='markers',
                        name=cluster_name,
                        marker=dict(
                            size=10,
                            color=colors[cluster_id % len(colors)],
                            line=dict(width=1, color='#ffffff'),
                            opacity=0.7
                        ),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. Cluster distribution (main + sub-clusters)
        if 'sub_cluster_label' in self.embeddings_df.columns:
            sub_cluster_counts = self.embeddings_df['sub_cluster_label'].value_counts().sort_index()
            sub_cluster_names_list = sub_cluster_counts.index.tolist()
            
            # Create color mapping
            color_map = {}
            for idx, (cluster_id, sub_id) in enumerate(self.embeddings_df[['cluster', 'sub_cluster']].drop_duplicates().values):
                color_map[f"{cluster_id}_{sub_id}"] = colors[cluster_id % len(colors)]
            
            bar_colors = []
            for label in sub_cluster_names_list:
                # Extract cluster_id from label or use default
                cluster_id = 0
                for cid in self.embeddings_df['cluster'].unique():
                    if self.embeddings_df[self.embeddings_df['cluster'] == cid]['sub_cluster_label'].iloc[0] == label:
                        cluster_id = cid
                        break
                bar_colors.append(colors[cluster_id % len(colors)])
            
            fig.add_trace(
                go.Bar(
                    x=sub_cluster_names_list,
                    y=sub_cluster_counts.values,
                    marker_color=bar_colors,
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=sub_cluster_counts.values,
                    textposition='auto',
                ),
                row=1, col=2
            )
        else:
            cluster_counts = self.embeddings_df['cluster'].value_counts().sort_index()
            cluster_names_list = [self.embeddings_df[self.embeddings_df['cluster'] == c]['cluster_label'].iloc[0] 
                                 for c in cluster_counts.index]
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names_list,
                    y=cluster_counts.values,
                    marker_color=[colors[i % len(colors)] for i in cluster_counts.index],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=cluster_counts.values,
                    textposition='auto',
                ),
                row=1, col=2
            )
        
        # 3. Transaction completion by sub-cluster
        if 'sub_cluster_label' in self.embeddings_df.columns:
            completion_by_sub = self.embeddings_df.groupby('sub_cluster_label')['completed_transaction'].agg(['mean', 'count'])
            sub_names = completion_by_sub.index.tolist()
            
            sub_bar_colors = []
            for label in sub_names:
                cluster_id = 0
                for cid in self.embeddings_df['cluster'].unique():
                    if self.embeddings_df[self.embeddings_df['cluster'] == cid]['sub_cluster_label'].iloc[0] == label:
                        cluster_id = cid
                        break
                sub_bar_colors.append(colors[cluster_id % len(colors)])
            
            fig.add_trace(
                go.Bar(
                    x=sub_names,
                    y=completion_by_sub['mean'] * 100,
                    marker_color=sub_bar_colors,
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=[f"{v:.1f}%" for v in completion_by_sub['mean'] * 100],
                    textposition='auto',
                ),
                row=2, col=1
            )
        else:
            completion_by_cluster = self.embeddings_df.groupby('cluster')['completed_transaction'].agg(['mean', 'count'])
            cluster_names_list = [self.embeddings_df[self.embeddings_df['cluster'] == c]['cluster_label'].iloc[0] 
                                 for c in completion_by_cluster.index]
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names_list,
                    y=completion_by_cluster['mean'] * 100,
                    marker_color=[colors[i % len(colors)] for i in completion_by_cluster.index],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=[f"{v:.1f}%" for v in completion_by_cluster['mean'] * 100],
                    textposition='auto',
                ),
                row=2, col=1
            )
        
        # 4. Sub-cluster comparison (replacing slow t-SNE with faster visualization)
        # Show sub-cluster sizes comparison
        if 'sub_cluster_label' in self.embeddings_df.columns:
            sub_cluster_sizes = self.embeddings_df['sub_cluster_label'].value_counts().sort_index()
            sub_names = sub_cluster_sizes.index.tolist()
            
            sub_bar_colors = []
            for label in sub_names:
                cluster_id = 0
                for cid in self.embeddings_df['cluster'].unique():
                    matching = self.embeddings_df[self.embeddings_df['cluster'] == cid]
                    if len(matching) > 0 and matching['sub_cluster_label'].iloc[0] == label:
                        cluster_id = cid
                        break
                sub_bar_colors.append(colors[cluster_id % len(colors)])
            
            fig.add_trace(
                go.Bar(
                    x=sub_names,
                    y=sub_cluster_sizes.values,
                    marker_color=sub_bar_colors,
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=sub_cluster_sizes.values,
                    textposition='auto',
                ),
                row=2, col=2
            )
        else:
            # Fallback: show main cluster sizes
            cluster_sizes = self.embeddings_df['cluster'].value_counts().sort_index()
            cluster_names_list = [self.embeddings_df[self.embeddings_df['cluster'] == c]['cluster_label'].iloc[0] 
                                 for c in cluster_sizes.index]
            
            fig.add_trace(
                go.Bar(
                    x=cluster_names_list,
                    y=cluster_sizes.values,
                    marker_color=[colors[i % len(colors)] for i in cluster_sizes.index],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=False,
                    text=cluster_sizes.values,
                    textposition='auto',
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="YUP Credit Card User Intent Clustering (Gemini Embedding + K-means)",
            title_x=0.5,
            title_font_size=20,
            title_font_color='#0a2540',
            showlegend=True,
            template="plotly_white",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f7f8fa',
            font=dict(family="Arial, sans-serif", size=11, color='#2d3748'),
        )
        
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="Sub-Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Number of Users", row=1, col=2)
        fig.update_xaxes(title_text="Sub-Cluster", row=2, col=1)
        fig.update_yaxes(title_text="Completion Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Sub-Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Users", row=2, col=2)
        
        return fig
    
    def generate_additional_cluster_visualizations(self):
        """Generate additional cluster visualizations"""
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        visualizations = {}
        
        # 1. Cluster Feature Heatmap
        visualizations['heatmap'] = self._generate_cluster_heatmap()
        
        # 2. Cluster Timeline Distribution
        visualizations['timeline'] = self._generate_cluster_timeline()
        
        # 3. Cluster Behavior Pattern Comparison
        visualizations['behavior_pattern'] = self._generate_behavior_pattern_comparison()
        
        # 4. Cluster Feature Radar Chart
        visualizations['radar'] = self._generate_cluster_radar()
        
        return visualizations
    
    def _generate_cluster_heatmap(self):
        """Generate heatmap showing feature comparison across clusters"""
        # Calculate features for each cluster
        cluster_features = []
        feature_names = []
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            cluster_df = self.df[self.df['user_uuid'].isin(cluster_users)]
            
            features = {
                'Completion Rate': cluster_data['completed_transaction'].mean() * 100,
                'Avg Events': cluster_df.groupby('user_uuid').size().mean(),
                'Behavior Diversity': cluster_df.groupby('user_uuid')['event_name'].nunique().mean(),
                'Unique Events': cluster_df['event_name'].nunique(),
                'Total Events': len(cluster_df),
            }
            
            if 'approved_time' in self.df.columns:
                cluster_df['approved_time'] = pd.to_datetime(cluster_df['approved_time'], errors='coerce')
                cluster_df['event_time'] = pd.to_datetime(cluster_df['event_time'], errors='coerce')
                time_to_first = []
                for user_id in cluster_users[:100]:  # Sample for performance
                    user_data = cluster_df[cluster_df['user_uuid'] == user_id]
                    if len(user_data) > 0 and pd.notna(user_data['approved_time'].iloc[0]):
                        approved = user_data['approved_time'].iloc[0]
                        first_event = user_data['event_time'].min()
                        if pd.notna(first_event):
                            time_to_first.append((first_event - approved).total_seconds() / 3600)
                if time_to_first:
                    features['Avg Time to First Action (hours)'] = np.mean(time_to_first)
                else:
                    features['Avg Time to First Action (hours)'] = 0
            
            if not feature_names:
                feature_names = list(features.keys())
            
            cluster_features.append([features.get(name, 0) for name in feature_names])
        
        # Normalize features for better visualization
        cluster_features = np.array(cluster_features)
        cluster_features_normalized = (cluster_features - cluster_features.min(axis=0)) / (cluster_features.max(axis=0) - cluster_features.min(axis=0) + 1e-10)
        
        cluster_labels = [f"Cluster {i} ({self.embeddings_df[self.embeddings_df['cluster'] == i]['cluster_label'].iloc[0]})" 
                         for i in sorted(self.embeddings_df['cluster'].unique())]
        
        fig = go.Figure(data=go.Heatmap(
            z=cluster_features_normalized.T,
            x=cluster_labels,
            y=feature_names,
            colorscale='Viridis',
            text=cluster_features.T,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Normalized Value")
        ))
        
        fig.update_layout(
            title="Cluster Feature Comparison Heatmap",
            xaxis_title="Cluster",
            yaxis_title="Feature",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def _generate_cluster_timeline(self):
        """Generate timeline distribution for clusters"""
        if 'approved_time' not in self.df.columns:
            return None
        
        self.df['approved_time'] = pd.to_datetime(self.df['approved_time'], errors='coerce')
        
        fig = go.Figure()
        
        colors = ['#2c5282', '#d4af37', '#48bb78', '#ed8936', '#9f7aea']
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            cluster_df = self.df[self.df['user_uuid'].isin(cluster_users)]
            
            approved_times = cluster_df['approved_time'].dropna()
            if len(approved_times) > 0:
                # Group by hour of day
                hours = approved_times.dt.hour
                hour_counts = hours.value_counts().sort_index()
                
                cluster_name = cluster_data['cluster_label'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    mode='lines+markers',
                    name=f"Cluster {cluster_id} ({cluster_name})",
                    line=dict(color=colors[cluster_id % len(colors)], width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="User Activation Time Distribution by Cluster",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Users",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    def _generate_behavior_pattern_comparison(self):
        """Generate behavior pattern comparison across clusters"""
        cluster_patterns = {}
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            cluster_df = self.df[self.df['user_uuid'].isin(cluster_users)]
            
            # Get top 10 most common events for this cluster
            event_counts = cluster_df['event_name'].value_counts().head(10)
            cluster_patterns[cluster_id] = event_counts.to_dict()
        
        # Create comparison chart
        all_events = set()
        for patterns in cluster_patterns.values():
            all_events.update(patterns.keys())
        
        fig = go.Figure()
        
        colors = ['#2c5282', '#d4af37', '#48bb78', '#ed8936', '#9f7aea']
        
        for cluster_id in sorted(cluster_patterns.keys()):
            cluster_name = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]['cluster_label'].iloc[0]
            event_counts = cluster_patterns[cluster_id]
            
            # Normalize by cluster size
            cluster_size = len(self.embeddings_df[self.embeddings_df['cluster'] == cluster_id])
            normalized_counts = {event: (count / cluster_size * 100) for event, count in event_counts.items()}
            
            top_events = sorted(normalized_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            events = [e[0] for e in top_events]
            counts = [e[1] for e in top_events]
            
            fig.add_trace(go.Bar(
                x=events,
                y=counts,
                name=f"Cluster {cluster_id}",
                marker_color=colors[cluster_id % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Top Behavior Patterns by Cluster (Normalized)",
            xaxis_title="Event Type",
            yaxis_title="Percentage of Users in Cluster",
            barmode='group',
            height=500,
            template="plotly_white",
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    def _generate_cluster_radar(self):
        """Generate radar chart comparing cluster features"""
        # Calculate normalized features for each cluster
        features_data = []
        feature_names = ['Completion Rate', 'Activity Level', 'Behavior Diversity', 'Engagement', 'Conversion Speed']
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            cluster_df = self.df[self.df['user_uuid'].isin(cluster_users)]
            
            completion_rate = cluster_data['completed_transaction'].mean()
            avg_events = cluster_df.groupby('user_uuid').size().mean()
            diversity = cluster_df.groupby('user_uuid')['event_name'].nunique().mean()
            
            # Normalize to 0-100 scale
            features = [
                completion_rate * 100,  # Completion Rate
                min(100, avg_events * 5),  # Activity Level
                min(100, diversity * 20),  # Behavior Diversity
                min(100, len(cluster_df) / len(cluster_users) * 2),  # Engagement
                completion_rate * 100  # Conversion Speed (same as completion for now)
            ]
            
            features_data.append(features)
        
        fig = go.Figure()
        
        colors = ['#2c5282', '#d4af37', '#48bb78', '#ed8936', '#9f7aea']
        
        for idx, cluster_id in enumerate(sorted(self.embeddings_df['cluster'].unique())):
            cluster_name = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]['cluster_label'].iloc[0]
            features = features_data[idx]
            
            fig.add_trace(go.Scatterpolar(
                r=features,
                theta=feature_names,
                fill='toself',
                name=f"Cluster {cluster_id} ({cluster_name})",
                line_color=colors[cluster_id % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Cluster Feature Radar Chart",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def generate_user_behavior_path(self, user_id):
        """Generate behavior path visualization for a single user"""
        if user_id not in self.df['user_uuid'].values:
            raise ValueError(f"User {user_id} not found in data")
        
        user_data = self.df[self.df['user_uuid'] == user_id].copy()
        user_data = user_data.sort_values('event_time')
        
        # Get cluster info
        if user_id in self.embeddings_df['user_uuid'].values:
            user_cluster_data = self.embeddings_df[self.embeddings_df['user_uuid'] == user_id].iloc[0]
            cluster_id = user_cluster_data['cluster']
            cluster_label = user_cluster_data['cluster_label']
        else:
            cluster_id = None
            cluster_label = "Unknown"
        
        visualizations = {}
        
        # 1. Timeline visualization
        visualizations['timeline'] = self._generate_user_timeline(user_data, user_id, cluster_label)
        
        # 2. Sankey diagram for event flow
        visualizations['sankey'] = self._generate_user_sankey(user_data, user_id)
        
        # 3. Event sequence network
        visualizations['network'] = self._generate_user_network(user_data, user_id)
        
        return visualizations, cluster_id, cluster_label
    
    def _generate_user_timeline(self, user_data, user_id, cluster_label):
        """Generate timeline visualization for user events"""
        if 'event_time' not in user_data.columns:
            return None
        
        user_data['event_time'] = pd.to_datetime(user_data['event_time'], errors='coerce')
        user_data = user_data.sort_values('event_time')
        
        # Create timeline
        events = user_data['event_name'].tolist()
        times = user_data['event_time'].tolist()
        
        # Color events by type
        event_colors = {}
        unique_events = list(set(events))
        colors = px.colors.qualitative.Set3
        for i, event in enumerate(unique_events):
            event_colors[event] = colors[i % len(colors)]
        
        fig = go.Figure()
        
        for i, (event, time) in enumerate(zip(events, times)):
            if pd.notna(time):
                fig.add_trace(go.Scatter(
                    x=[time, time],
                    y=[i, i+1],
                    mode='lines+markers',
                    name=event,
                    line=dict(color=event_colors.get(event, '#999999'), width=3),
                    marker=dict(size=10, color=event_colors.get(event, '#999999')),
                    showlegend=False,
                    hovertemplate=f'<b>{event}</b><br>Time: %{{x}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"User {user_id[:8]}... Behavior Timeline (Cluster: {cluster_label})",
            xaxis_title="Time",
            yaxis_title="Event Sequence",
            height=400,
            template="plotly_white",
            hovermode='closest'
        )
        
        return fig
    
    def _generate_user_sankey(self, user_data, user_id):
        """Generate Sankey diagram for user event flow"""
        events = user_data['event_name'].tolist()
        
        if len(events) < 2:
            return None
        
        # Build transition counts
        transitions = {}
        for i in range(len(events) - 1):
            source = events[i]
            target = events[i + 1]
            key = (source, target)
            transitions[key] = transitions.get(key, 0) + 1
        
        # Create node and link lists
        all_nodes = list(set(events))
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        sources = []
        targets = []
        values = []
        labels = []
        
        for (source, target), count in transitions.items():
            sources.append(node_indices[source])
            targets.append(node_indices[target])
            values.append(count)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color="#2c5282"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(44, 82, 130, 0.4)"
            )
        )])
        
        fig.update_layout(
            title=f"User {user_id[:8]}... Event Flow (Sankey Diagram)",
            font_size=10,
            height=500
        )
        
        return fig
    
    def _generate_user_network(self, user_data, user_id):
        """Generate network graph for user behavior sequence"""
        events = user_data['event_name'].tolist()
        
        if len(events) < 2:
            return None
        
        # Build edge list
        edges = []
        for i in range(len(events) - 1):
            edges.append((events[i], events[i + 1]))
        
        # Count edge weights
        edge_weights = {}
        for edge in edges:
            edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # Create network visualization using scatter plot
        all_nodes = list(set(events))
        node_positions = {}
        angle_step = 2 * np.pi / len(all_nodes)
        for i, node in enumerate(all_nodes):
            angle = i * angle_step
            node_positions[node] = (np.cos(angle), np.sin(angle))
        
        fig = go.Figure()
        
        # Draw edges
        for (source, target), weight in edge_weights.items():
            x0, y0 = node_positions[source]
            x1, y1 = node_positions[target]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=weight*2, color='rgba(44, 82, 130, 0.3)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Draw nodes
        node_x = [node_positions[node][0] for node in all_nodes]
        node_y = [node_positions[node][1] for node in all_nodes]
        node_sizes = [events.count(node) * 10 for node in all_nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color='#2c5282', line=dict(width=2, color='white')),
            text=all_nodes,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            showlegend=False,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"User {user_id[:8]}... Behavior Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def generate_user_path_page(self, user_id):
        """Generate a standalone HTML page for a single user's behavior path"""
        if user_id not in self.df['user_uuid'].values:
            return None
        
        visualizations, cluster_id, cluster_label = self.generate_user_behavior_path(user_id)
        
        # Convert visualizations to HTML
        timeline_html = visualizations['timeline'].to_html(include_plotlyjs='cdn', div_id='timeline-chart', full_html=False) if visualizations.get('timeline') else None
        sankey_html = visualizations['sankey'].to_html(include_plotlyjs=False, div_id='sankey-chart', full_html=False) if visualizations.get('sankey') else None
        network_html = visualizations['network'].to_html(include_plotlyjs=False, div_id='network-chart', full_html=False) if visualizations.get('network') else None
        
        # Get user data summary
        user_data = self.df[self.df['user_uuid'] == user_id].copy()
        user_data = user_data.sort_values('event_time')
        total_events = len(user_data)
        unique_events = user_data['event_name'].nunique()
        has_transaction = user_data['fir_trx_time'].notna().any() if 'fir_trx_time' in user_data.columns else False
        
        if 'event_time' in user_data.columns and len(user_data) > 0:
            user_data['event_time'] = pd.to_datetime(user_data['event_time'], errors='coerce')
            first_event = user_data['event_time'].min()
            last_event = user_data['event_time'].max()
            if pd.notna(first_event) and pd.notna(last_event):
                duration = (last_event - first_event).total_seconds() / 3600  # hours
            else:
                duration = 0
        else:
            duration = 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Behavior Path - {user_id[:20]}...</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
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
        }}
        .header {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 50%, #2c5282 100%);
            color: #ffffff;
            padding: 40px 50px;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .header-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .info-card {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 6px;
        }}
        .info-card .label {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .info-card .value {{
            font-size: 1.3em;
            font-weight: 700;
        }}
        .content {{
            padding: 40px 50px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #0a2540;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #d4af37;
        }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .back-link {{
            display: inline-block;
            padding: 10px 20px;
            background: #2c5282;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        .back-link:hover {{
            background: #1a365d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="back-link">← Back to Main Report</a>
            <h1>User Behavior Path Analysis</h1>
            <p style="opacity: 0.9; margin-top: 10px;">User ID: {user_id}</p>
            <div class="header-info">
                <div class="info-card">
                    <div class="label">Cluster</div>
                    <div class="value">Cluster {cluster_id}: {cluster_label}</div>
                </div>
                <div class="info-card">
                    <div class="label">Total Events</div>
                    <div class="value">{total_events}</div>
                </div>
                <div class="info-card">
                    <div class="label">Unique Event Types</div>
                    <div class="value">{unique_events}</div>
                </div>
                <div class="info-card">
                    <div class="label">Session Duration</div>
                    <div class="value">{duration:.1f} hours</div>
                </div>
                <div class="info-card">
                    <div class="label">Transaction Status</div>
                    <div class="value">{'Completed' if has_transaction else 'Not Completed'}</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <!-- Timeline Visualization -->
            {f'''
            <div class="section">
                <h2>📅 Behavior Timeline</h2>
                <p style="color: #718096; margin-bottom: 15px;">
                    Chronological sequence of all user events showing the progression of user actions over time.
                </p>
                <div class="chart-container">
                    {timeline_html}
                </div>
            </div>
            ''' if timeline_html else ''}
            
            <!-- Sankey Diagram -->
            {f'''
            <div class="section">
                <h2>🌊 Event Flow (Sankey Diagram)</h2>
                <p style="color: #718096; margin-bottom: 15px;">
                    Visual representation of event transitions showing how the user moves between different actions.
                </p>
                <div class="chart-container">
                    {sankey_html}
                </div>
            </div>
            ''' if sankey_html else ''}
            
            <!-- Network Graph -->
            {f'''
            <div class="section">
                <h2>🕸️ Behavior Network</h2>
                <p style="color: #718096; margin-bottom: 15px;">
                    Network visualization showing the relationships and connections between different user actions.
                </p>
                <div class="chart-container">
                    {network_html}
                </div>
            </div>
            ''' if network_html else ''}
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _get_available_gemini_model(self):
        """Get an available Gemini model for text generation"""
        # Cache the model if already found
        if hasattr(self, '_cached_text_model'):
            return self._cached_text_model
        
        # If we already tried and failed, don't try again
        if hasattr(self, '_text_model_failed'):
            return None
        
        # First, try to list available models
        available_models = []
        try:
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    available_models.append(model_name)
        except Exception as e:
            # If listing fails, use default list
            pass
        
        # Try different model names in order of preference
        # If we got available models from API, use those first
        if available_models:
            model_names = available_models + ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            # Remove duplicates while preserving order
            model_names = list(dict.fromkeys(model_names))
        else:
            # Fallback to common model names
            model_names = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro',
                'gemini-1.0-pro',
            ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test with a minimal prompt to verify it works
                try:
                    test_response = model.generate_content("Hi", generation_config={'max_output_tokens': 1})
                    if test_response and test_response.text:
                        self._cached_text_model = model
                        self._cached_model_name = model_name
                        if not hasattr(self, '_model_announced'):
                            print(f"   ✅ Using Gemini text model: {model_name}")
                            self._model_announced = True
                        return model
                except Exception as test_e:
                    # Model exists but test failed, try next one
                    continue
            except Exception as e:
                # Model doesn't exist, try next one
                continue
        
        # If no model works, mark as failed and return None
        self._text_model_failed = True
        if not hasattr(self, '_model_announced'):
            print(f"   ⚠️  No available Gemini text model found, will use fallback text generation")
            self._model_announced = True
        return None
    
    def _call_gemini_text(self, prompt, max_retries=3, base_delay=2):
        """Call Gemini API to generate text content
        
        Args:
            prompt: The prompt for text generation
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
        """
        model = self._get_available_gemini_model()
        
        if model is None:
            return None
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"   ⚠️  Empty response from Gemini API (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in ['500', '503', '429', 'rate limit', 'quota', 'internal error'])
                
                # If model not found error, try to get a different model
                if ('not found' in error_msg or 'not supported' in error_msg) and attempt == 0:
                    # Clear cache and try to get a different model
                    if hasattr(self, '_cached_text_model'):
                        delattr(self, '_cached_text_model')
                    model = self._get_available_gemini_model()
                    if model is None:
                        print(f"   ⚠️  No available model, using fallback text generation")
                        break
                    continue
                
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"   ⚠️  Retryable error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   ⏳ Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ Error calling Gemini API: {e}")
                    if not is_retryable:
                        break
        
        return None
    
    def _calculate_business_metrics(self):
        """Calculate business-oriented metrics for each cluster"""
        business_metrics = {}
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            
            # Get user behavior data
            cluster_df = self.df[self.df['user_uuid'].isin(cluster_users)].copy()
            
            metrics = {}
            
            # 1. User value tier
            completion_rate = cluster_data['completed_transaction'].mean()
            avg_events = cluster_df.groupby('user_uuid').size().mean()
            
            if completion_rate >= 0.7:
                value_tier = "High Value"
                value_color = "#48bb78"
            elif completion_rate >= 0.3:
                value_tier = "Medium Value"
                value_color = "#d4af37"
            else:
                value_tier = "Low Value"
                value_color = "#e53e3e"
            
            metrics['value_tier'] = value_tier
            metrics['value_color'] = value_color
            metrics['completion_rate'] = completion_rate * 100
            metrics['avg_events'] = avg_events
            
            # 2. User lifecycle stage
            if 'approved_time' in self.df.columns:
                cluster_df['approved_time'] = pd.to_datetime(cluster_df['approved_time'], errors='coerce')
                cluster_df['event_time'] = pd.to_datetime(cluster_df['event_time'], errors='coerce')
                
                # Calculate time from activation to first action
                user_lifecycle = []
                for user_id in cluster_users:
                    user_data = cluster_df[cluster_df['user_uuid'] == user_id]
                    if len(user_data) > 0 and pd.notna(user_data['approved_time'].iloc[0]):
                        approved = user_data['approved_time'].iloc[0]
                        first_event = user_data['event_time'].min()
                        if pd.notna(first_event):
                            time_to_first = (first_event - approved).total_seconds() / 3600  # hours
                            
                            if time_to_first < 1:
                                lifecycle = "New User (Active within 1 hour)"
                            elif time_to_first < 24:
                                lifecycle = "Active User (Active within 24 hours)"
                            elif time_to_first < 168:  # 7 days
                                lifecycle = "Regular User (Active within 7 days)"
                            else:
                                lifecycle = "At-Risk User (Inactive for 7+ days)"
                            
                            user_lifecycle.append(lifecycle)
                
                if user_lifecycle:
                    lifecycle_counts = pd.Series(user_lifecycle).value_counts()
                    metrics['lifecycle_stage'] = lifecycle_counts.index[0]  # Most common stage
                    metrics['lifecycle_distribution'] = lifecycle_counts.to_dict()
                else:
                    metrics['lifecycle_stage'] = "Unknown"
                    metrics['lifecycle_distribution'] = {}
            else:
                metrics['lifecycle_stage'] = "Unknown"
                metrics['lifecycle_distribution'] = {}
            
            # 3. Behavior feature analysis
            event_types = cluster_df['event_name'].value_counts()
            top_events = event_types.head(5).to_dict()
            
            # Calculate behavior diversity
            unique_events_per_user = cluster_df.groupby('user_uuid')['event_name'].nunique().mean()
            metrics['behavior_diversity'] = unique_events_per_user
            metrics['top_events'] = top_events
            
            # 4. Conversion path analysis
            if 'fir_trx_time' in cluster_df.columns:
                users_with_tx = cluster_df[cluster_df['fir_trx_time'].notna()]['user_uuid'].unique()
                if len(users_with_tx) > 0:
                    # Analyze behavior paths of users who completed transactions
                    tx_users_data = cluster_df[cluster_df['user_uuid'].isin(users_with_tx)]
                    tx_path = tx_users_data.groupby('user_uuid')['event_name'].apply(list)
                    
                    # Find most common conversion paths
                    path_lengths = tx_path.apply(len)
                    metrics['avg_path_length'] = path_lengths.mean()
                    metrics['conversion_users'] = len(users_with_tx)
                else:
                    metrics['avg_path_length'] = 0
                    metrics['conversion_users'] = 0
            else:
                metrics['avg_path_length'] = 0
                metrics['conversion_users'] = 0
            
            business_metrics[cluster_id] = metrics
        
        return business_metrics
    
    def _generate_operational_recommendations(self, cluster_id, cluster_name, metrics, completion_rate):
        """Generate operational recommendations for each cluster using Gemini API"""
        # Prepare context for Gemini
        context = f"""
Cluster Analysis Data:
- Cluster ID: {cluster_id}
- Cluster Name: {cluster_name}
- Transaction Completion Rate: {completion_rate:.1f}%
- User Value Tier: {metrics.get('value_tier', 'Unknown')}
- Average Events per User: {metrics.get('avg_events', 0):.1f}
- Behavior Diversity: {metrics.get('behavior_diversity', 0):.1f} unique event types
- Conversion Users: {metrics.get('conversion_users', 0)}
- Lifecycle Stage: {metrics.get('lifecycle_stage', 'Unknown')}
- Top 5 Events: {', '.join([f'{k} ({v} times)' for k, v in list(metrics.get('top_events', {}).items())[:5]])}

Please generate operational recommendations for this user cluster. Return the response in JSON format with the following structure:
{{
    "recommendations": [
        {{
            "priority": "High/Medium/Low",
            "title": "Recommendation title",
            "description": "Detailed description",
            "actions": ["Action 1", "Action 2", "Action 3"]
        }}
    ]
}}

Generate 2-4 recommendations based on the cluster characteristics. Focus on actionable, business-oriented suggestions.
"""
        
        try:
            response_text = self._call_gemini_text(context, max_retries=3, base_delay=2)
            if response_text:
                # Try to parse JSON from response
                import json
                import re
                
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                else:
                    # Try to find JSON object directly
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0)
                
                try:
                    result = json.loads(response_text)
                    if 'recommendations' in result:
                        return result['recommendations']
                except json.JSONDecodeError:
                    print(f"   ⚠️  Failed to parse JSON from Gemini response, using fallback")
        except Exception as e:
            print(f"   ⚠️  Error generating recommendations with Gemini: {e}, using fallback")
        
        # Fallback to simple rule-based recommendations
        recommendations = []
        if completion_rate >= 0.7:
            recommendations.append({
                'priority': 'High',
                'title': 'Maintain Premium Experience',
                'description': 'This cluster has high conversion rate and represents core users. Maintain current premium experience and consider recommending related value-added services.',
                'actions': [
                    'Optimize transaction flow for smooth experience',
                    'Provide personalized recommendations for cross-selling',
                    'Establish VIP service system to enhance loyalty'
                ]
            })
        elif completion_rate >= 0.3:
            recommendations.append({
                'priority': 'Medium',
                'title': 'Improve Conversion Rate',
                'description': 'This cluster shows conversion potential but needs guidance. Optimize product presentation and simplify processes.',
                'actions': [
                    'Optimize product display pages, highlight core value',
                    'Simplify transaction flow, reduce steps',
                    'Offer limited-time promotions or new user exclusives',
                    'Enhance customer support for timely assistance'
                ]
            })
        else:
            recommendations.append({
                'priority': 'High',
                'title': 'Activate Users',
                'description': 'This cluster has low conversion rate and needs focused activation efforts. Guide users to complete first transaction through multiple channels.',
                'actions': [
                    'Send personalized push notifications to remind users',
                    'Provide onboarding guides and tutorials',
                    'Design incentive mechanisms like first transaction rewards',
                    'Analyze user churn reasons and optimize experience'
                ]
            })
        
        return recommendations
    
    def _generate_user_portrait(self, cluster_id, cluster_name, metrics, completion_rate):
        """Generate user portrait description using Gemini API"""
        # Prepare context for Gemini
        context = f"""
Cluster Analysis Data:
- Cluster ID: {cluster_id}
- Cluster Name: {cluster_name}
- Transaction Completion Rate: {completion_rate:.1f}%
- User Value Tier: {metrics.get('value_tier', 'Unknown')}
- Average Events per User: {metrics.get('avg_events', 0):.1f}
- Behavior Diversity: {metrics.get('behavior_diversity', 0):.1f} unique event types
- Conversion Users: {metrics.get('conversion_users', 0)} out of total users
- Lifecycle Stage: {metrics.get('lifecycle_stage', 'Unknown')}
- Top 5 Events: {', '.join([f'{k} ({v} times)' for k, v in list(metrics.get('top_events', {}).items())[:5]])}
- Lifecycle Distribution: {metrics.get('lifecycle_distribution', {})}

Please generate a user portrait description for this cluster in plain, business-friendly English. 
The description should be 3-5 sentences, easy to understand, and help business operators quickly grasp the characteristics of this user group.
Focus on: conversion behavior, activity level, feature exploration, and lifecycle stage.
Do not use markdown formatting, just plain text.
"""
        
        try:
            response_text = self._call_gemini_text(context, max_retries=3, base_delay=2)
            if response_text:
                # Clean up the response (remove markdown, extra whitespace)
                response_text = response_text.strip()
                # Remove markdown formatting if present
                response_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response_text)  # Remove bold
                response_text = re.sub(r'\*([^*]+)\*', r'\1', response_text)  # Remove italic
                return response_text
        except Exception as e:
            print(f"   ⚠️  Error generating user portrait with Gemini: {e}, using fallback")
        
        # Fallback to simple rule-based description
        portraits = []
        if completion_rate >= 0.7:
            portraits.append("This is a group of high-conversion users who typically complete their first transaction soon after account activation, showing strong purchase intent.")
        elif completion_rate >= 0.3:
            portraits.append("This is a group of medium-conversion users who show interest in the product but may still be hesitating or comparing options.")
        else:
            portraits.append("This is a group of low-conversion users who have activated their accounts but haven't completed a transaction yet, and may need more guidance.")
        
        avg_events = metrics.get('avg_events', 0)
        if avg_events > 20:
            portraits.append("They are very active, performing many operations on the platform, indicating high interest in the product.")
        elif avg_events > 10:
            portraits.append("They are moderately active with regular usage frequency, but may need more guidance.")
        else:
            portraits.append("They have relatively few actions, possibly new users or not yet familiar with the product.")
        
        diversity = metrics.get('behavior_diversity', 0)
        if diversity >= 5:
            portraits.append("They have explored multiple features, showing comprehensive understanding of product capabilities.")
        elif diversity >= 3:
            portraits.append("They have used some core features, showing basic product awareness.")
        else:
            portraits.append("Their behavior is relatively single-focused, possibly only engaging with specific features.")
        
        return " ".join(portraits)
    
    def generate_report(self):
        """Generate HTML report"""
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        fig = self.generate_visualizations()
        plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='main-chart', full_html=False)
        
        # Generate additional cluster visualizations
        print("   📊 Generating additional cluster visualizations...")
        additional_viz = self.generate_additional_cluster_visualizations()
        
        # Convert additional visualizations to HTML
        additional_viz_html = {}
        for viz_name, viz_fig in additional_viz.items():
            if viz_fig is not None:
                additional_viz_html[viz_name] = viz_fig.to_html(include_plotlyjs=False, div_id=f'{viz_name}-chart', full_html=False)
            else:
                additional_viz_html[viz_name] = None
        
        report_time = datetime.now().strftime("%B %d, %Y %H:%M")
        
        # Calculate cluster statistics
        cluster_stats = []
        sub_cluster_stats = []
        
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_name = cluster_data['cluster_label'].iloc[0]
            completion_rate = cluster_data['completed_transaction'].mean() * 100
            
            cluster_stats.append({
                'id': cluster_id,
                'name': cluster_name,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(self.embeddings_df) * 100,
                'completion_rate': completion_rate
            })
            
            # Calculate sub-cluster statistics
            if 'sub_cluster' in self.embeddings_df.columns:
                for sub_id in sorted(cluster_data['sub_cluster'].unique()):
                    sub_data = cluster_data[cluster_data['sub_cluster'] == sub_id]
                    if len(sub_data) > 0:
                        sub_label = sub_data['sub_cluster_label'].iloc[0]
                        sub_completion = sub_data['completed_transaction'].mean() * 100
                        
                        sub_cluster_stats.append({
                            'main_cluster_id': cluster_id,
                            'main_cluster_name': cluster_name,
                            'sub_id': sub_id,
                            'sub_label': sub_label,
                            'count': len(sub_data),
                            'percentage': len(sub_data) / len(self.embeddings_df) * 100,
                            'completion_rate': sub_completion
                        })
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YUP User Intent Clustering - Gemini Embedding Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
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
        }}
        .header {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 50%, #2c5282 100%);
            color: #ffffff;
            padding: 60px 50px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.4em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .content {{
            padding: 50px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section h2 {{
            color: #0a2540;
            font-size: 1.75em;
            margin-bottom: 25px;
            padding-bottom: 12px;
            border-bottom: 2px solid #d4af37;
        }}
        .card {{
            background: #f7f8fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2c5282;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #0a2540;
            margin: 10px 0;
        }}
        .summary-card .label {{
            color: #718096;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #0a2540;
            color: white;
            font-weight: 600;
        }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YUP Credit Card User Intent Clustering</h1>
            <p style="font-size: 1.1em; opacity: 0.9; margin-top: 10px;">
                Analysis using Google Gemini Embedding Model + K-means Clustering
            </p>
            <p style="font-size: 0.9em; opacity: 0.8; margin-top: 15px;">
                Report Generated: {report_time}
            </p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Summary</h2>
                <div class="summary-cards">
                    <div class="summary-card">
                        <div class="label">Total Users</div>
                        <div class="value">{len(self.embeddings_df)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Main Clusters</div>
                        <div class="value">{len(cluster_stats)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Sub-Clusters</div>
                        <div class="value">{len(sub_cluster_stats) if sub_cluster_stats else len(cluster_stats)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Users with Transaction</div>
                        <div class="value">{self.embeddings_df['completed_transaction'].sum()}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Embedding Dimensions</div>
                        <div class="value">768</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Main Cluster Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Label</th>
                            <th>Users</th>
                            <th>Percentage</th>
                            <th>Transaction Completion Rate</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for stat in cluster_stats:
            html_content += f"""
                        <tr>
                            <td><strong>Cluster {stat['id']}</strong></td>
                            <td>{stat['name']}</td>
                            <td>{stat['count']}</td>
                            <td>{stat['percentage']:.1f}%</td>
                            <td>{stat['completion_rate']:.1f}%</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        # Add business-oriented analysis sections
        html_content += """
            <div class="section">
                <h2>💼 Business Scenario Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.05em;">
                    The following analysis provides business-oriented insights to help you better understand different user groups and develop targeted operational strategies.
                </p>
"""
        
        # Add business analysis for each cluster
        for cluster_stat in cluster_stats:
            cluster_id = cluster_stat['id']
            cluster_name = cluster_stat['name']
            completion_rate = cluster_stat['completion_rate']
            metrics = business_metrics.get(cluster_id, {})
            
            # Generate user portrait
            user_portrait = self._generate_user_portrait(cluster_id, cluster_name, metrics, completion_rate)
            
            # Generate recommendations
            recommendations = self._generate_operational_recommendations(cluster_id, cluster_name, metrics, completion_rate / 100)
            
            html_content += f"""
                <div class="card" style="margin-bottom: 40px; border-left: 4px solid {'#2c5282' if cluster_id == 0 else '#d4af37'};">
                    <h3 style="color: #0a2540; margin-bottom: 20px; font-size: 1.5em;">
                        {cluster_name} (Cluster {cluster_id})
                    </h3>
                    
                    <!-- User Portrait -->
                    <div style="background: #f7f8fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: #2d3748; margin-bottom: 15px; font-size: 1.2em;">👤 User Portrait</h4>
                        <p style="color: #4a5568; line-height: 1.8; font-size: 1.05em;">
                            {user_portrait}
                        </p>
                    </div>
                    
                    <!-- Key Metrics -->
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                        <div style="background: #e6fffa; padding: 15px; border-radius: 6px; border-left: 4px solid #48bb78;">
                            <div style="color: #234e52; font-size: 0.9em; margin-bottom: 5px;">User Value Tier</div>
                            <div style="color: #0a2540; font-size: 1.3em; font-weight: 700;">{metrics.get('value_tier', 'Unknown')}</div>
                        </div>
                        <div style="background: #feebc8; padding: 15px; border-radius: 6px; border-left: 4px solid #d4af37;">
                            <div style="color: #7c2d12; font-size: 0.9em; margin-bottom: 5px;">Avg Events per User</div>
                            <div style="color: #0a2540; font-size: 1.3em; font-weight: 700;">{metrics.get('avg_events', 0):.1f}</div>
                        </div>
                        <div style="background: #e6f3ff; padding: 15px; border-radius: 6px; border-left: 4px solid #2c5282;">
                            <div style="color: #2c5282; font-size: 0.9em; margin-bottom: 5px;">Behavior Diversity</div>
                            <div style="color: #0a2540; font-size: 1.3em; font-weight: 700;">{metrics.get('behavior_diversity', 0):.1f} types</div>
                        </div>
                        <div style="background: #f0f4ff; padding: 15px; border-radius: 6px; border-left: 4px solid #9f7aea;">
                            <div style="color: #553c9a; font-size: 0.9em; margin-bottom: 5px;">Conversion Users</div>
                            <div style="color: #0a2540; font-size: 1.3em; font-weight: 700;">{metrics.get('conversion_users', 0)}</div>
                        </div>
                    </div>
                    
                    <!-- 生命周期分布 -->
"""
            if metrics.get('lifecycle_distribution'):
                html_content += """
                    <div style="background: #fff5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: #2d3748; margin-bottom: 15px; font-size: 1.2em;">📅 User Lifecycle Distribution</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">
"""
                for stage, count in metrics.get('lifecycle_distribution', {}).items():
                    html_content += f"""
                            <div style="background: white; padding: 12px; border-radius: 6px; border-left: 3px solid #2c5282;">
                                <div style="color: #4a5568; font-size: 0.85em;">{stage}</div>
                                <div style="color: #0a2540; font-size: 1.1em; font-weight: 600;">{count} users</div>
                            </div>
"""
                html_content += """
                        </div>
                    </div>
"""
            
            # 热门行为
            if metrics.get('top_events'):
                html_content += """
                    <div style="background: #f0fff4; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="color: #2d3748; margin-bottom: 15px; font-size: 1.2em;">🔥 Top 5 Popular Behaviors</h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
"""
                for event, count in list(metrics.get('top_events', {}).items())[:5]:
                    html_content += f"""
                            <div style="background: white; padding: 8px 15px; border-radius: 6px; border: 1px solid #c6f6d5;">
                                <span style="color: #22543d; font-weight: 600;">{event}</span>
                                <span style="color: #718096; font-size: 0.9em; margin-left: 8px;">{count} times</span>
                            </div>
"""
                html_content += """
                        </div>
                    </div>
"""
            
            html_content += """
                    
                    <!-- Operational Recommendations -->
                    <div style="background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%); padding: 25px; border-radius: 8px; border: 2px solid #2c5282;">
                        <h4 style="color: #0a2540; margin-bottom: 20px; font-size: 1.3em; font-weight: 700;">💡 Operational Strategy Recommendations</h4>
"""
            
            for rec in recommendations:
                priority_color = "#e53e3e" if rec['priority'] == 'High' else "#d4af37" if rec['priority'] == 'Medium' else "#48bb78"
                html_content += f"""
                        <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid {priority_color};">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <span style="background: {priority_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em; font-weight: 600;">
                                    {rec['priority']} Priority
                                </span>
                                <h5 style="color: #0a2540; font-size: 1.1em; margin: 0;">{rec['title']}</h5>
                            </div>
                            <p style="color: #4a5568; line-height: 1.8; margin-bottom: 12px;">
                                {rec['description']}
                            </p>
                            <div style="background: #f7f8fa; padding: 15px; border-radius: 6px;">
                                <div style="color: #2d3748; font-weight: 600; margin-bottom: 10px; font-size: 0.95em;">Actionable Recommendations:</div>
                                <ul style="color: #4a5568; line-height: 2; padding-left: 20px; margin: 0;">
"""
                for action in rec['actions']:
                    html_content += f"""
                                    <li>{action}</li>
"""
                html_content += """
                                </ul>
                            </div>
                        </div>
"""
            
            html_content += """
                    </div>
                </div>
"""
        
        html_content += """
            </div>
            
            <!-- Conversion Funnel Analysis -->
            <div class="section">
                <h2>📊 Conversion Funnel Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.05em;">
                    This section shows the complete conversion path from account activation to transaction completion, helping identify conversion bottlenecks.
                </p>
                <div style="background: #f7f8fa; padding: 30px; border-radius: 8px;">
"""
        
        # Calculate conversion funnel
        total_users = len(self.embeddings_df)
        activated_users = total_users
        active_users = len(self.embeddings_df[self.embeddings_df['completed_transaction'] == 1])
        conversion_rate = (active_users / total_users * 100) if total_users > 0 else 0
        
        # Generate conversion insights using Gemini
        conversion_insight_prompt = f"""
Conversion Funnel Data:
- Total Activated Users: {activated_users}
- Users with First Transaction: {active_users}
- Overall Conversion Rate: {conversion_rate:.1f}%
- Non-converted Users: {total_users - active_users} ({100 - conversion_rate:.1f}%)

Please generate 2-3 concise insights about this conversion funnel in plain English. 
Focus on: conversion rate assessment, growth opportunities, and strategic recommendations.
Return only the insights as a list, one per line, without numbering or markdown.
"""
        
        conversion_insights = self._call_gemini_text(conversion_insight_prompt, max_retries=2, base_delay=1)
        if not conversion_insights:
            # Fallback insights
            if conversion_rate >= 30:
                insight1 = f"Overall conversion rate is {conversion_rate:.1f}%, which is above industry average."
            elif conversion_rate >= 10:
                insight1 = f"Overall conversion rate is {conversion_rate:.1f}%, showing significant room for improvement."
            else:
                insight1 = f"Overall conversion rate is {conversion_rate:.1f}%, requiring immediate attention."
            insight2 = f"There are {total_users - active_users} users ({100 - conversion_rate:.1f}%) who haven't completed their first transaction, representing potential growth opportunities."
            insight3 = "Recommend developing differentiated conversion strategies for different user groups to improve overall conversion rate."
            conversion_insights = f"{insight1}\n{insight2}\n{insight3}"
        
        html_content += f"""
                    <div style="display: flex; flex-direction: column; gap: 20px; max-width: 600px; margin: 0 auto;">
                        <div style="background: linear-gradient(135deg, #0a2540 0%, #1a365d 100%); color: white; padding: 25px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 8px;">Account Activated</div>
                            <div style="font-size: 2.5em; font-weight: 700;">{activated_users}</div>
                            <div style="font-size: 0.85em; opacity: 0.8; margin-top: 5px;">100%</div>
                        </div>
                        
                        <div style="text-align: center; color: #718096; font-size: 1.2em;">↓</div>
                        
                        <div style="background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%); color: white; padding: 25px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.9em; opacity: 0.9; margin-bottom: 8px;">First Transaction Completed</div>
                            <div style="font-size: 2.5em; font-weight: 700;">{active_users}</div>
                            <div style="font-size: 0.85em; opacity: 0.8; margin-top: 5px;">{conversion_rate:.1f}%</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 20px; background: white; border-radius: 8px; border-left: 4px solid #d4af37;">
                        <h4 style="color: #0a2540; margin-bottom: 15px;">💡 Conversion Insights</h4>
                        <ul style="color: #4a5568; line-height: 2; padding-left: 20px;">
"""
        # Split insights and add as list items
        for insight in conversion_insights.strip().split('\n'):
            if insight.strip():
                html_content += f"""
                            <li>{insight.strip()}</li>
"""
        html_content += """
                        </ul>
                    </div>
                </div>
            </div>
"""
        
        # Add detailed sub-cluster analysis for each main cluster
        if sub_cluster_stats:
            html_content += """
            <div class="section">
                <h2>🔬 Detailed Sub-Cluster Analysis by Main Cluster</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.05em;">
                    This section provides detailed analysis of sub-clusters within each main cluster, 
                    showing how users are further segmented based on their behavioral patterns.
                </p>
"""
            
            # Group sub-clusters by main cluster
            for cluster_stat in cluster_stats:
                cluster_id = cluster_stat['id']
                cluster_name = cluster_stat['name']
                
                # Get sub-clusters for this main cluster
                cluster_sub_stats = [s for s in sub_cluster_stats if s['main_cluster_id'] == cluster_id]
                
                if cluster_sub_stats:
                    html_content += f"""
                <div class="card" style="margin-bottom: 30px; border-left: 4px solid {'#2c5282' if cluster_id == 0 else '#d4af37'};">
                    <h3 style="color: #0a2540; margin-bottom: 20px; font-size: 1.3em;">
                        {cluster_name} (Cluster {cluster_id})
                    </h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        This main cluster contains {len(cluster_sub_stats)} sub-clusters with {cluster_stat['count']} users total.
                    </p>
                    <table style="margin-top: 15px;">
                        <thead>
                            <tr>
                                <th>Sub-Cluster</th>
                                <th>Label</th>
                                <th>Users</th>
                                <th>% of Main Cluster</th>
                                <th>% of Total Users</th>
                                <th>Transaction Completion Rate</th>
                            </tr>
                        </thead>
                        <tbody>
"""
                    
                    for sub_stat in cluster_sub_stats:
                        pct_of_main = (sub_stat['count'] / cluster_stat['count'] * 100) if cluster_stat['count'] > 0 else 0
                        html_content += f"""
                            <tr>
                                <td><strong>Sub-{sub_stat['sub_id']}</strong></td>
                                <td>{sub_stat['sub_label'].split(' - ')[-1] if ' - ' in sub_stat['sub_label'] else sub_stat['sub_label']}</td>
                                <td>{sub_stat['count']}</td>
                                <td>{pct_of_main:.1f}%</td>
                                <td>{sub_stat['percentage']:.1f}%</td>
                                <td style="color: {'#48bb78' if sub_stat['completion_rate'] > 50 else '#e53e3e'}; font-weight: 600;">
                                    {sub_stat['completion_rate']:.1f}%
                                </td>
                            </tr>
"""
                    
                    html_content += """
                        </tbody>
                    </table>
                    
                    <!-- Sub-cluster insights -->
                    <div style="margin-top: 20px; padding: 15px; background: #f7f8fa; border-radius: 6px;">
                        <h4 style="color: #2d3748; margin-bottom: 12px; font-size: 1.05em;">Key Insights:</h4>
                        <ul style="color: #4a5568; line-height: 1.8; padding-left: 20px;">
"""
                    
                    # Generate insights
                    max_completion_sub = max(cluster_sub_stats, key=lambda x: x['completion_rate'])
                    min_completion_sub = min(cluster_sub_stats, key=lambda x: x['completion_rate'])
                    largest_sub = max(cluster_sub_stats, key=lambda x: x['count'])
                    
                    html_content += f"""
                            <li>The <strong>{max_completion_sub['sub_label'].split(' - ')[-1]}</strong> sub-cluster has the highest transaction completion rate ({max_completion_sub['completion_rate']:.1f}%)</li>
                            <li>The <strong>{largest_sub['sub_label'].split(' - ')[-1]}</strong> sub-cluster is the largest, containing {largest_sub['count']} users ({largest_sub['count']/cluster_stat['count']*100:.1f}% of this main cluster)</li>
                            <li>The <strong>{min_completion_sub['sub_label'].split(' - ')[-1]}</strong> sub-cluster shows the lowest completion rate ({min_completion_sub['completion_rate']:.1f}%), indicating potential conversion opportunities</li>
"""
                    
                    html_content += """
                        </ul>
                    </div>
                </div>
"""
            
            html_content += """
            </div>
"""
        
        if sub_cluster_stats:
            html_content += """
            <div class="section">
                <h2>🔬 Sub-Cluster Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 20px;">
                    Detailed breakdown of sub-clusters within each main cluster:
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Main Cluster</th>
                            <th>Sub-Cluster Label</th>
                            <th>Users</th>
                            <th>Percentage</th>
                            <th>Transaction Completion Rate</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            
            for stat in sub_cluster_stats:
                html_content += f"""
                        <tr>
                            <td><strong>{stat['main_cluster_name']}</strong></td>
                            <td>{stat['sub_label']}</td>
                            <td>{stat['count']}</td>
                            <td>{stat['percentage']:.1f}%</td>
                            <td>{stat['completion_rate']:.1f}%</td>
                        </tr>
"""
            
            html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        html_content += """
            <div class="section">
                <h2>📈 Main Visualizations</h2>
                <div class="chart-container">
                    PLOTLY_CHART_PLACEHOLDER
                </div>
            </div>
            
            <!-- Additional Cluster Visualizations -->
            <div class="section">
                <h2>📊 Additional Cluster Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.05em;">
                    Advanced visualizations providing deeper insights into cluster characteristics and patterns.
                </p>
"""
        
        # Add additional visualizations
        if additional_viz_html.get('heatmap'):
            html_content += f"""
                <div class="chart-container" style="margin-bottom: 30px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px; font-size: 1.3em;">Cluster Feature Comparison Heatmap</h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        Compare normalized features across all clusters to identify key differences.
                    </p>
                    {additional_viz_html['heatmap']}
                </div>
"""
        
        if additional_viz_html.get('timeline'):
            html_content += f"""
                <div class="chart-container" style="margin-bottom: 30px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px; font-size: 1.3em;">User Activation Time Distribution</h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        Analyze when users from different clusters activate their accounts throughout the day.
                    </p>
                    {additional_viz_html['timeline']}
                </div>
"""
        
        if additional_viz_html.get('behavior_pattern'):
            html_content += f"""
                <div class="chart-container" style="margin-bottom: 30px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px; font-size: 1.3em;">Behavior Pattern Comparison</h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        Compare the most common behaviors across different clusters (normalized by cluster size).
                    </p>
                    {additional_viz_html['behavior_pattern']}
                </div>
"""
        
        if additional_viz_html.get('radar'):
            html_content += f"""
                <div class="chart-container" style="margin-bottom: 30px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px; font-size: 1.3em;">Cluster Feature Radar Chart</h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        Multi-dimensional comparison of cluster features using a radar chart.
                    </p>
                    {additional_viz_html['radar']}
                </div>
"""
        
        html_content += """
            </div>
            
            <!-- Single User Behavior Path Analysis -->
            <div class="section">
                <h2>🔍 Single User Behavior Path Analysis</h2>
                <p style="color: #4a5568; margin-bottom: 25px; font-size: 1.05em;">
                    Explore individual user behavior paths. Select a user to view their detailed behavior timeline, event flow, and network visualization.
                </p>
                
                <div style="background: #f7f8fa; padding: 25px; border-radius: 8px; margin-bottom: 25px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px; font-size: 1.2em;">User Selection</h3>
                    <p style="color: #718096; margin-bottom: 15px; font-size: 0.95em;">
                        Select a user from the list below to view their detailed behavior path. Users are grouped by cluster for easy navigation.
                    </p>
                    <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-bottom: 15px;">
                        <label for="user-select" style="color: #4a5568; font-weight: 600;">Select User:</label>
                        <select id="user-select" style="padding: 10px 15px; border: 2px solid #e2e8f0; border-radius: 6px; font-size: 0.95em; min-width: 300px; background: white;">
                            <option value="">-- Select a user --</option>
"""
        
        # Add user options grouped by cluster and sub-cluster
        for cluster_id in sorted(self.embeddings_df['cluster'].unique()):
            cluster_data = self.embeddings_df[self.embeddings_df['cluster'] == cluster_id]
            cluster_name = cluster_data['cluster_label'].iloc[0]
            
            # Group by sub-cluster
            if 'sub_cluster' in cluster_data.columns:
                for sub_cluster_id in sorted(cluster_data['sub_cluster'].unique()):
                    sub_cluster_data = cluster_data[cluster_data['sub_cluster'] == sub_cluster_id]
                    sub_cluster_users = sub_cluster_data['user_uuid'].unique()
                    
                    # Get sub-cluster name if available
                    if sub_cluster_id >= 0 and len(sub_cluster_data) > 0:
                        sub_cluster_label = sub_cluster_data['sub_cluster_label'].iloc[0] if 'sub_cluster_label' in sub_cluster_data.columns else f"Sub-cluster {sub_cluster_id}"
                        optgroup_label = f"Cluster {cluster_id} ({cluster_name}) - {sub_cluster_label} ({len(sub_cluster_users)} users)"
                    else:
                        optgroup_label = f"Cluster {cluster_id} ({cluster_name}) - No sub-cluster ({len(sub_cluster_users)} users)"
                    
                    html_content += f"""
                            <optgroup label="{optgroup_label}">
"""
                    for user_id in sub_cluster_users:
                        html_content += f"""
                                <option value="{user_id}">{user_id[:25]}...</option>
"""
                    html_content += """
                            </optgroup>
"""
            else:
                # Fallback if no sub-cluster column
                cluster_users = cluster_data['user_uuid'].unique()[:30]
                html_content += f"""
                            <optgroup label="Cluster {cluster_id}: {cluster_name} ({len(cluster_users)} users shown)">
"""
                for user_id in cluster_users[:15]:
                    html_content += f"""
                                <option value="{user_id}">{user_id[:25]}...</option>
"""
                html_content += """
                            </optgroup>
"""
        
        html_content += """
                        </select>
                        <button onclick="loadUserPath()" style="padding: 10px 25px; background: #2c5282; color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: all 0.3s ease;">
                            View User Path
                        </button>
                    </div>
                    <p style="color: #718096; font-size: 0.85em; font-style: italic; margin-top: 10px;">
                        Click "View User Path" to open a detailed page with timeline, Sankey diagram, and network visualizations for the selected user.
                    </p>
                </div>
                
                <script type="text/javascript">
                    // Define loadUserPath function in global scope - must be defined before button onclick
                    function loadUserPath() {{
                        const select = document.getElementById('user-select');
                        if (!select) {{
                            console.error('User select element not found');
                            return;
                        }}
                        const userId = select.value;
                        
                        if (!userId) {{
                            alert('Please select a user first');
                            return;
                        }}
                        
                        // Sanitize user_id for filename (same as Python code)
                        const safeUserId = userId.replace(/[^a-zA-Z0-9._-]/g, '_');
                        const userPathUrl = 'user_path_' + safeUserId + '.html';
                        
                        // Try to open the static page first
                        // If it doesn't exist, fallback to interactive demo
                        fetch(userPathUrl, {{ method: 'HEAD' }})
                            .then(function(response) {{
                                if (response.ok) {{
                                    // File exists, open it
                                    window.open(userPathUrl, '_blank');
                                }} else {{
                                    // File doesn't exist, open interactive demo with user ID
                                    const demoUrl = 'intent_demo.html?user=' + encodeURIComponent(userId);
                                    window.open(demoUrl, '_blank');
                                }}
                            }})
                            .catch(function() {{
                                // On error (e.g., CORS or file doesn't exist), open interactive demo
                                const demoUrl = 'intent_demo.html?user=' + encodeURIComponent(userId);
                                window.open(demoUrl, '_blank');
                            }});
                    }}
                    
                    // Ensure function is available globally
                    window.loadUserPath = loadUserPath;
                </script>
            </div>
            
            <div class="section">
                <h2>🔬 Methodology</h2>
                <div class="card">
                    <h3 style="color: #0a2540; margin-bottom: 15px;">Feature Extraction</h3>
                    <p style="color: #4a5568; line-height: 1.8;">
                        User behavior sequences and campaign information are converted to text representations,
                        which are then embedded using Google Gemini's embedding-001 model (768 dimensions).
                        No manual feature engineering is performed - the model automatically learns
                        semantic representations of user behavior patterns.
                    </p>
                </div>
                <div class="card" style="margin-top: 20px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px;">Clustering Algorithm</h3>
                    <p style="color: #4a5568; line-height: 1.8;">
                        K-means clustering is applied to the standardized embedding vectors.
                        The optimal number of main clusters is automatically determined using the silhouette score.
                        After identifying main clusters, hierarchical sub-clustering is performed within each main cluster
                        to identify finer-grained user segments. This two-level approach allows for both broad user intent
                        categories and detailed behavioral patterns within each category.
                    </p>
                </div>
                <div class="card" style="margin-top: 20px;">
                    <h3 style="color: #0a2540; margin-bottom: 15px;">Sub-Clustering Strategy</h3>
                    <p style="color: #4a5568; line-height: 1.8;">
                        For each main cluster, sub-clustering is performed independently using K-means on the embedding vectors
                        of users within that cluster. The optimal number of sub-clusters (typically 2-4) is determined using
                        silhouette score analysis. Sub-clusters are automatically named based on their transaction completion
                        rates, helping identify high-value segments and conversion opportunities.
                    </p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        html_content = html_content.replace('PLOTLY_CHART_PLACEHOLDER', plotly_html)
        return html_content
    
    def generate_user_action_cluster_mapping(self):
        """Generate mapping of users, their actions, and clusters"""
        if self.embeddings_df is None or 'cluster' not in self.embeddings_df.columns:
            return None
        
        # Merge clustering results with original data
        user_cluster_map = self.embeddings_df[['user_uuid', 'cluster', 'cluster_label', 'sub_cluster', 'sub_cluster_label']].copy()
        
        # Get user actions from original data
        user_actions = []
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            
            # Get cluster info
            cluster_info = user_cluster_map[user_cluster_map['user_uuid'] == user_id]
            if len(cluster_info) > 0:
                cluster_id = cluster_info['cluster'].iloc[0]
                cluster_label = cluster_info['cluster_label'].iloc[0]
                sub_cluster = cluster_info['sub_cluster'].iloc[0] if 'sub_cluster' in cluster_info.columns else None
                sub_cluster_label = cluster_info['sub_cluster_label'].iloc[0] if 'sub_cluster_label' in cluster_info.columns else None
            else:
                cluster_id = None
                cluster_label = "未分类"
                sub_cluster = None
                sub_cluster_label = None
            
            # Get unique actions
            unique_actions = user_data['event_name'].unique().tolist()
            action_counts = user_data['event_name'].value_counts().to_dict()
            
            # Get transaction status
            has_transaction = pd.notna(user_data['fir_trx_time'].iloc[0]) if 'fir_trx_time' in user_data.columns else False
            
            user_actions.append({
                'user_uuid': user_id,
                'cluster_id': int(cluster_id) if cluster_id is not None else None,
                'cluster_label': cluster_label,
                'sub_cluster_id': int(sub_cluster) if sub_cluster is not None else None,
                'sub_cluster_label': sub_cluster_label,
                'total_actions': len(user_data),
                'unique_actions': unique_actions,
                'action_counts': action_counts,
                'has_transaction': bool(has_transaction),
                'first_action': user_data['event_name'].iloc[0] if len(user_data) > 0 else None,
                'last_action': user_data['event_name'].iloc[-1] if len(user_data) > 0 else None
            })
        
        return user_actions
    
    def generate_user_action_cluster_page(self):
        """Generate HTML page showing user actions and their clusters"""
        print("\n📄 Generating user-action-cluster mapping page...")
        
        user_actions = self.generate_user_action_cluster_mapping()
        if user_actions is None:
            return None
        
        report_time = datetime.now().strftime("%B %d, %Y %H:%M")
        
        # Prepare data for JavaScript
        user_data_json = json.dumps(user_actions, ensure_ascii=False, default=str)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>用户动作-聚类映射表 | YUP聚类分析</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Microsoft YaHei', Arial, sans-serif;
            background: #f0f2f5;
            padding: 30px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(10, 37, 64, 0.12);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #0a2540 0%, #1a365d 50%, #2c5282 100%);
            color: #ffffff;
            padding: 40px 50px;
        }}
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .header-actions {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .search-filter {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .search-box {{
            padding: 10px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 0.95em;
            min-width: 250px;
        }}
        .filter-select {{
            padding: 10px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 0.95em;
            background: white;
            cursor: pointer;
        }}
        .nav-link {{
            color: #ffffff;
            text-decoration: none;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 6px;
            transition: all 0.3s ease;
        }}
        .nav-link:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .content {{
            padding: 30px 50px;
        }}
        .stats-bar {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: #f7f8fa;
            padding: 15px 25px;
            border-radius: 6px;
            border-left: 4px solid #2c5282;
        }}
        .stat-card .label {{
            color: #718096;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .stat-card .value {{
            color: #0a2540;
            font-size: 1.5em;
            font-weight: 700;
        }}
        .user-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin-top: 20px;
        }}
        .user-table th {{
            background: #0a2540;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .user-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .user-table tr:hover {{
            background: #f7f8fa;
        }}
        .cluster-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .cluster-0 {{
            background: #c6f6d5;
            color: #22543d;
        }}
        .cluster-1 {{
            background: #feebc8;
            color: #7c2d12;
        }}
        .cluster-2 {{
            background: #bee3f8;
            color: #2c5282;
        }}
        .sub-cluster-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 8px;
            font-size: 0.75em;
            margin-left: 5px;
            background: #e2e8f0;
            color: #4a5568;
        }}
        .action-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .action-tag {{
            background: #edf2f7;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #4a5568;
        }}
        .action-tag.with-count {{
            background: #e6fffa;
            color: #234e52;
        }}
        .transaction-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .transaction-yes {{
            background: #c6f6d5;
            color: #22543d;
        }}
        .transaction-no {{
            background: #fed7d7;
            color: #742a2a;
        }}
        .pagination {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 30px;
        }}
        .pagination button {{
            padding: 8px 15px;
            border: 1px solid #e2e8f0;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .pagination button:hover {{
            background: #f7f8fa;
            border-color: #2c5282;
        }}
        .pagination button.active {{
            background: #2c5282;
            color: white;
            border-color: #2c5282;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>用户动作-聚类映射表</h1>
            <p style="opacity: 0.9; margin-top: 5px;">查看每个用户的行为动作及其所属聚类</p>
            <p style="font-size: 0.85em; opacity: 0.8; margin-top: 10px;">报告生成时间: {report_time}</p>
            <div class="header-actions">
                <div class="search-filter">
                    <input type="text" id="searchInput" class="search-box" placeholder="搜索用户ID或动作...">
                    <select id="clusterFilter" class="filter-select">
                        <option value="">所有聚类</option>
                    </select>
                    <select id="transactionFilter" class="filter-select">
                        <option value="">所有用户</option>
                        <option value="yes">已完成交易</option>
                        <option value="no">未完成交易</option>
                    </select>
                </div>
                <a href="index.html" class="nav-link">← 返回主报告</a>
            </div>
        </div>
        
        <div class="content">
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="label">总用户数</div>
                    <div class="value" id="totalUsers">{len(user_actions)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">显示用户数</div>
                    <div class="value" id="displayedUsers">{len(user_actions)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">聚类数量</div>
                    <div class="value" id="clusterCount">{len(set([ua['cluster_id'] for ua in user_actions if ua['cluster_id'] is not None]))}</div>
                </div>
            </div>
            
            <table class="user-table" id="userTable">
                <thead>
                    <tr>
                        <th>用户ID</th>
                        <th>主聚类</th>
                        <th>子聚类</th>
                        <th>交易状态</th>
                        <th>动作总数</th>
                        <th>唯一动作</th>
                        <th>首次动作</th>
                        <th>最后动作</th>
                    </tr>
                </thead>
                <tbody id="userTableBody">
                    <!-- Data will be populated by JavaScript -->
                </tbody>
            </table>
            
            <div class="pagination" id="pagination">
                <!-- Pagination will be generated by JavaScript -->
            </div>
        </div>
    </div>
    
    <script>
        // User data
        const userData = {user_data_json};
        
        // Current state
        let filteredData = [...userData];
        let currentPage = 1;
        const itemsPerPage = 50;
        
        // Initialize
        function init() {{
            populateClusterFilter();
            renderTable();
            setupEventListeners();
        }}
        
        function populateClusterFilter() {{
            const clusterFilter = document.getElementById('clusterFilter');
            const clusters = [...new Set(userData.map(u => u.cluster_label).filter(Boolean))];
            clusters.forEach(cluster => {{
                const option = document.createElement('option');
                option.value = cluster;
                option.textContent = cluster;
                clusterFilter.appendChild(option);
            }});
        }}
        
        function setupEventListeners() {{
            document.getElementById('searchInput').addEventListener('input', handleFilter);
            document.getElementById('clusterFilter').addEventListener('change', handleFilter);
            document.getElementById('transactionFilter').addEventListener('change', handleFilter);
        }}
        
        function handleFilter() {{
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const clusterFilter = document.getElementById('clusterFilter').value;
            const transactionFilter = document.getElementById('transactionFilter').value;
            
            filteredData = userData.filter(user => {{
                // Search filter
                const matchesSearch = !searchTerm || 
                    user.user_uuid.toLowerCase().includes(searchTerm) ||
                    user.unique_actions.some(action => action.toLowerCase().includes(searchTerm));
                
                // Cluster filter
                const matchesCluster = !clusterFilter || user.cluster_label === clusterFilter;
                
                // Transaction filter
                const matchesTransaction = !transactionFilter ||
                    (transactionFilter === 'yes' && user.has_transaction) ||
                    (transactionFilter === 'no' && !user.has_transaction);
                
                return matchesSearch && matchesCluster && matchesTransaction;
            }});
            
            currentPage = 1;
            updateStats();
            renderTable();
        }}
        
        function updateStats() {{
            document.getElementById('displayedUsers').textContent = filteredData.length;
        }}
        
        function renderTable() {{
            const tbody = document.getElementById('userTableBody');
            tbody.innerHTML = '';
            
            const startIndex = (currentPage - 1) * itemsPerPage;
            const endIndex = startIndex + itemsPerPage;
            const pageData = filteredData.slice(startIndex, endIndex);
            
            pageData.forEach(user => {{
                const row = document.createElement('tr');
                
                // User ID
                const userIdCell = document.createElement('td');
                userIdCell.textContent = user.user_uuid.substring(0, 12) + '...';
                userIdCell.style.fontFamily = 'monospace';
                userIdCell.style.fontSize = '0.9em';
                
                // Cluster
                const clusterCell = document.createElement('td');
                if (user.cluster_id !== null) {{
                    const badge = document.createElement('span');
                    badge.className = `cluster-badge cluster-${{user.cluster_id}}`;
                    badge.textContent = `${{user.cluster_label}}`;
                    clusterCell.appendChild(badge);
                    
                    if (user.sub_cluster_label) {{
                        const subBadge = document.createElement('span');
                        subBadge.className = 'sub-cluster-badge';
                        subBadge.textContent = user.sub_cluster_label;
                        clusterCell.appendChild(subBadge);
                    }}
                }} else {{
                    clusterCell.textContent = '未分类';
                }}
                
                // Sub-cluster
                const subClusterCell = document.createElement('td');
                if (user.sub_cluster_label) {{
                    subClusterCell.textContent = user.sub_cluster_label;
                }} else {{
                    subClusterCell.textContent = '-';
                    subClusterCell.style.color = '#a0aec0';
                }}
                
                // Transaction status
                const transactionCell = document.createElement('td');
                const transBadge = document.createElement('span');
                transBadge.className = `transaction-badge transaction-${{user.has_transaction ? 'yes' : 'no'}}`;
                transBadge.textContent = user.has_transaction ? '已完成' : '未完成';
                transactionCell.appendChild(transBadge);
                
                // Total actions
                const totalCell = document.createElement('td');
                totalCell.textContent = user.total_actions;
                
                // Unique actions
                const actionsCell = document.createElement('td');
                const actionTags = document.createElement('div');
                actionTags.className = 'action-tags';
                user.unique_actions.slice(0, 5).forEach(action => {{
                    const tag = document.createElement('span');
                    tag.className = 'action-tag';
                    const count = user.action_counts[action] || 0;
                    tag.textContent = count > 1 ? `${{action}} (${{count}})` : action;
                    if (count > 1) tag.className += ' with-count';
                    actionTags.appendChild(tag);
                }});
                if (user.unique_actions.length > 5) {{
                    const moreTag = document.createElement('span');
                    moreTag.className = 'action-tag';
                    moreTag.textContent = `+${{user.unique_actions.length - 5}} 更多`;
                    actionTags.appendChild(moreTag);
                }}
                actionsCell.appendChild(actionTags);
                
                // First action
                const firstCell = document.createElement('td');
                firstCell.textContent = user.first_action || '-';
                firstCell.style.fontSize = '0.9em';
                
                // Last action
                const lastCell = document.createElement('td');
                lastCell.textContent = user.last_action || '-';
                lastCell.style.fontSize = '0.9em';
                
                row.appendChild(userIdCell);
                row.appendChild(clusterCell);
                row.appendChild(subClusterCell);
                row.appendChild(transactionCell);
                row.appendChild(totalCell);
                row.appendChild(actionsCell);
                row.appendChild(firstCell);
                row.appendChild(lastCell);
                
                tbody.appendChild(row);
            }});
            
            renderPagination();
        }}
        
        function renderPagination() {{
            const pagination = document.getElementById('pagination');
            pagination.innerHTML = '';
            
            const totalPages = Math.ceil(filteredData.length / itemsPerPage);
            
            if (totalPages <= 1) return;
            
            // Previous button
            const prevBtn = document.createElement('button');
            prevBtn.textContent = '上一页';
            prevBtn.disabled = currentPage === 1;
            prevBtn.onclick = () => {{
                if (currentPage > 1) {{
                    currentPage--;
                    renderTable();
                }}
            }};
            pagination.appendChild(prevBtn);
            
            // Page numbers
            for (let i = 1; i <= totalPages; i++) {{
                if (i === 1 || i === totalPages || (i >= currentPage - 2 && i <= currentPage + 2)) {{
                    const pageBtn = document.createElement('button');
                    pageBtn.textContent = i;
                    pageBtn.className = i === currentPage ? 'active' : '';
                    pageBtn.onclick = () => {{
                        currentPage = i;
                        renderTable();
                    }};
                    pagination.appendChild(pageBtn);
                }} else if (i === currentPage - 3 || i === currentPage + 3) {{
                    const ellipsis = document.createElement('span');
                    ellipsis.textContent = '...';
                    ellipsis.style.padding = '8px';
                    pagination.appendChild(ellipsis);
                }}
            }}
            
            // Next button
            const nextBtn = document.createElement('button');
            nextBtn.textContent = '下一页';
            nextBtn.disabled = currentPage === totalPages;
            nextBtn.onclick = () => {{
                if (currentPage < totalPages) {{
                    currentPage++;
                    renderTable();
                }}
            }};
            pagination.appendChild(nextBtn);
        }}
        
        // Initialize on load
        init();
    </script>
</body>
</html>
"""
        
        return html_content


def main():
    """Main function"""
    print("🚀 Starting YUP credit card user behavior analysis...")
    print("   Using: Google Gemini Embedding + K-means Clustering")
    print("   No manual feature engineering\n")
    
    # Get API key
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Please set it with: export GEMINI_API_KEY='your-api-key'")
        return None
    
    # Initialize analyzer
    data_path = 'data.csv'
    try:
        analyzer = UserIntentAnalyzer(data_path, gemini_api_key=gemini_api_key)
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return None
    
    # Extract embeddings
    print("\n📊 Extracting embeddings using Gemini...")
    embeddings_df = analyzer.extract_embeddings()
    print(f"✅ Successfully extracted embeddings for {len(embeddings_df)} users\n")
    
    # Perform clustering
    print("🔍 Performing K-means clustering...")
    cluster_labels = analyzer.perform_clustering(n_clusters=None, max_k=10)
    num_clusters = len(set(cluster_labels))
    print(f"✅ Clustering completed: {num_clusters} clusters identified\n")
    
    # Show cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("📊 Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        percentage = count / len(cluster_labels) * 100
        cluster_name = analyzer.embeddings_df[analyzer.embeddings_df['cluster'] == cluster_id]['cluster_label'].iloc[0]
        print(f"   Cluster {cluster_id} ({cluster_name}): {count} users ({percentage:.1f}%)")
    
    # Generate main report (overall clustering)
    print("\n📄 Generating main HTML report...")
    html_content = analyzer.generate_report()
    
    # Save main report files
    output_path = 'index.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also save to user_intent_clustering_report.html for compatibility
    report_path = 'user_intent_clustering_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Main report saved to: {output_path} and {report_path}")
    
    # Generate user path pages for ALL users in each sub-cluster
    print("\n📊 Generating user behavior path pages for all users...")
    total_users = len(analyzer.embeddings_df)
    users_generated = 0
    users_failed = 0
    
    # Group by cluster and sub-cluster
    if 'sub_cluster' in analyzer.embeddings_df.columns:
        for cluster_id in sorted(analyzer.embeddings_df['cluster'].unique()):
            cluster_data = analyzer.embeddings_df[analyzer.embeddings_df['cluster'] == cluster_id]
            cluster_name = cluster_data['cluster_label'].iloc[0]
            
            for sub_cluster_id in sorted(cluster_data['sub_cluster'].unique()):
                sub_cluster_data = cluster_data[cluster_data['sub_cluster'] == sub_cluster_id]
                sub_cluster_users = sub_cluster_data['user_uuid'].unique()
                
                if sub_cluster_id >= 0:
                    sub_cluster_label = sub_cluster_data['sub_cluster_label'].iloc[0] if 'sub_cluster_label' in sub_cluster_data.columns else f"Sub-cluster {sub_cluster_id}"
                    print(f"   Generating paths for Cluster {cluster_id} ({cluster_name}) - {sub_cluster_label}: {len(sub_cluster_users)} users")
                else:
                    print(f"   Generating paths for Cluster {cluster_id} ({cluster_name}) - No sub-cluster: {len(sub_cluster_users)} users")
                
                for idx, user_id in enumerate(sub_cluster_users, 1):
                    try:
                        user_path_html = analyzer.generate_user_path_page(user_id)
                        if user_path_html:
                            # Sanitize user_id for filename (same as JavaScript)
                            safe_user_id = ''.join(c if c.isalnum() or c in '._-' else '_' for c in user_id)
                            user_path_filename = f"user_path_{safe_user_id}.html"
                            
                            with open(user_path_filename, 'w', encoding='utf-8') as f:
                                f.write(user_path_html)
                            users_generated += 1
                            
                            # Progress indicator every 50 users
                            if users_generated % 50 == 0:
                                print(f"      Progress: {users_generated}/{total_users} users generated...")
                    except Exception as e:
                        users_failed += 1
                        if users_failed <= 5:  # Only print first 5 errors
                            print(f"      ⚠️  Error generating path for user {user_id[:20]}...: {e}")
                        continue
    else:
        # Fallback if no sub-cluster column
        print("   ⚠️  No sub-cluster information found, generating for all users by cluster")
        for cluster_id in sorted(analyzer.embeddings_df['cluster'].unique()):
            cluster_data = analyzer.embeddings_df[analyzer.embeddings_df['cluster'] == cluster_id]
            cluster_users = cluster_data['user_uuid'].unique()
            
            for user_id in cluster_users:
                try:
                    user_path_html = analyzer.generate_user_path_page(user_id)
                    if user_path_html:
                        safe_user_id = ''.join(c if c.isalnum() or c in '._-' else '_' for c in user_id)
                        user_path_filename = f"user_path_{safe_user_id}.html"
                        
                        with open(user_path_filename, 'w', encoding='utf-8') as f:
                            f.write(user_path_html)
                        users_generated += 1
                except Exception as e:
                    users_failed += 1
                    continue
    
    print(f"✅ Generated {users_generated} user behavior path pages")
    if users_failed > 0:
        print(f"⚠️  Failed to generate {users_failed} user path pages")
    
    print(f"\n✅ All analysis completed!")
    print(f"📁 Main report: {output_path}")
    if users_generated > 0:
        print(f"📁 User path pages: {users_generated} pages generated")
    
    return analyzer


if __name__ == '__main__':
    analyzer = main()

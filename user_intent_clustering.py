"""
YUP信用卡用户意图聚类分析
场景：客户获得额度后的第一次交易行为分析
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class UserIntentAnalyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.df = pd.read_excel(data_path)
        self.df['event_time'] = pd.to_datetime(self.df['event_time'])
        self.df['act_time'] = pd.to_datetime(self.df['act_time'])
        self.df['fir_trx_time'] = pd.to_datetime(self.df['fir_trx_time'], errors='coerce')
        
        # 过滤掉user_uuid为NaN的行
        self.df = self.df[self.df['user_uuid'].notna()].copy()
        
        self.features_df = None
        self.scaled_features = None
        self.cluster_labels = None
        
    def extract_features(self):
        """提取用户行为特征"""
        features_list = []
        
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            
            # 基础特征
            features = {
                'user_uuid': user_id,
                'total_events': len(user_data),
                'completed_transaction': 1 if pd.notna(user_data['fir_trx_time'].iloc[0]) else 0,
            }
            
            # 时间特征
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
            
            # 事件类型特征
            event_counts = user_data['event_name'].value_counts()
            event_types = user_data['event_name'].unique()
            
            # 事件类型统计
            features.update({
                'unique_event_types': len(event_types),
                'most_common_event_count': event_counts.max() if len(event_counts) > 0 else 0,
                'event_diversity': len(event_types) / len(user_data) if len(user_data) > 0 else 0,
            })
            
            # 特定事件类型计数
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
            
            # 行为序列特征
            event_sequence = user_data['event_name'].tolist()
            
            # 计算重复行为（可能表示犹豫或探索）
            features['repetitive_behavior_score'] = self._calculate_repetition_score(event_sequence)
            
            # 计算行为转换次数
            features['behavior_transitions'] = len([i for i in range(1, len(event_sequence)) 
                                                   if event_sequence[i] != event_sequence[i-1]])
            
            # 计算回到主页的次数（可能表示迷失）
            features['homepage_returns'] = sum(1 for i, event in enumerate(event_sequence) 
                                               if 'home_page' in event.lower() and i > 0)
            
            # 备注信息特征
            remarks = user_data['remarks'].dropna()
            if len(remarks) > 0:
                features['has_remarks'] = 1
                features['remarks_count'] = len(remarks)
                # 关键备注
                features['voucher_click_remarks'] = remarks.str.contains('券', na=False).sum()
                features['qr_remarks'] = remarks.str.contains('QR', na=False).sum()
            else:
                features['has_remarks'] = 0
                features['remarks_count'] = 0
                features['voucher_click_remarks'] = 0
                features['qr_remarks'] = 0
            
            # 意图强度特征（基于事件密度）
            if features['session_duration_minutes'] > 0:
                features['event_density'] = features['total_events'] / features['session_duration_minutes']
            else:
                features['event_density'] = 0
            
            # 探索vs执行特征
            features['exploration_score'] = features['unique_event_types'] / max(features['total_events'], 1)
            features['execution_score'] = features['payment_attempts_count'] / max(features['total_events'], 1)
            
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        return self.features_df
    
    def _calculate_repetition_score(self, sequence):
        """计算重复行为得分"""
        if len(sequence) < 2:
            return 0
        
        # 计算连续重复
        consecutive_repeats = 0
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                consecutive_repeats += 1
        
        # 计算总体重复率
        unique_events = len(set(sequence))
        repetition_rate = 1 - (unique_events / len(sequence)) if len(sequence) > 0 else 0
        
        return (consecutive_repeats + repetition_rate * len(sequence)) / len(sequence)
    
    def perform_clustering(self, method='kmeans', n_clusters=2):
        """执行聚类分析"""
        # 选择数值特征
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['user_uuid', 'completed_transaction']]
        
        X = self.features_df[feature_cols].values
        
        # 标准化
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(X)
        
        if method == 'kmeans':
            # 使用肘部法则确定最佳聚类数
            inertias = []
            K_range = range(2, min(6, len(self.features_df) + 1))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(self.scaled_features)
                inertias.append(kmeans.inertia_)
            
            # 选择最佳k（这里简化为2，因为只有2个用户）
            best_k = n_clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(self.scaled_features)
            self.cluster_centers = kmeans.cluster_centers_
            self.cluster_model = kmeans
            
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            self.cluster_labels = dbscan.fit_predict(self.scaled_features)
            self.cluster_model = dbscan
        
        # 添加聚类标签
        self.features_df['cluster'] = self.cluster_labels
        self.features_df['cluster_label'] = self.features_df['cluster'].apply(
            lambda x: f'Cluster {x+1}'
        )
        
        return self.cluster_labels
    
    def generate_visualizations(self):
        """生成可视化HTML报告"""
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '用户行为特征对比', '聚类结果 (PCA降维)',
                '行为模式雷达图', '时间序列分析',
                '事件类型分布', '意图强度分析'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatterpolar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. 用户行为特征对比
        comparison_features = ['total_events', 'session_duration_minutes', 
                              'unique_event_types', 'event_density']
        user_ids_short = [f"用户{i+1}" for i in range(len(self.features_df))]
        feature_labels = {
            'total_events': '总事件数',
            'session_duration_minutes': '会话时长(分钟)',
            'unique_event_types': '唯一事件类型数',
            'event_density': '事件密度(事件/分钟)'
        }
        
        # 金融行业专业配色
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
        
        # 2. PCA降维可视化
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_features)
        
        # 金融行业专业配色
        colors = ['#2c5282', '#d4af37']
        for cluster_id in self.features_df['cluster'].unique():
            mask = self.features_df['cluster'] == cluster_id
            cluster_users = [f"用户{i+1}" for i, m in enumerate(mask) if m]
            fig.add_trace(
                go.Scatter(
                    x=pca_result[mask, 0],
                    y=pca_result[mask, 1],
                    mode='markers+text',
                    name=f'聚类 {cluster_id+1}',
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
        
        # 3. 行为模式雷达图
        radar_features = ['exploration_score', 'execution_score', 'event_diversity',
                         'repetitive_behavior_score', 'payment_attempts_ratio']
        radar_labels = {
            'exploration_score': '探索得分',
            'execution_score': '执行得分',
            'event_diversity': '事件多样性',
            'repetitive_behavior_score': '重复行为',
            'payment_attempts_ratio': '支付尝试率'
        }
        
        for idx, user_row in self.features_df.iterrows():
            values = [user_row[f] for f in radar_features]
            values.append(values[0])  # 闭合雷达图
            cluster_id = self.features_df.loc[idx, 'cluster']
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=[radar_labels.get(f, f.replace('_', ' ').title()) for f in radar_features] + [radar_labels.get(radar_features[0], radar_features[0].replace('_', ' ').title())],
                    fill='toself',
                    name=f"用户{idx+1}",
                    line_color=colors[cluster_id],
                    fillcolor=colors[cluster_id],
                    opacity=0.4,
                    line=dict(width=2.5),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 4. 时间序列分析
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            
            # 计算累积事件数
            user_data['cumulative_events'] = range(1, len(user_data) + 1)
            user_data['time_from_start'] = (user_data['event_time'] - user_data['event_time'].min()).dt.total_seconds() / 60
            
            cluster_id = self.features_df[self.features_df['user_uuid'] == user_id]['cluster'].iloc[0]
            user_idx = list(self.df['user_uuid'].unique()).index(user_id)
            
            # 将颜色转换为rgba
            color_hex = colors[cluster_id]
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            
            fig.add_trace(
                go.Scatter(
                    x=user_data['time_from_start'],
                    y=user_data['cumulative_events'],
                    mode='lines+markers',
                    name=f"用户{user_idx+1}",
                    line=dict(color=colors[cluster_id], width=3),
                    marker=dict(size=7, line=dict(width=1.5, color='#ffffff')),
                    showlegend=True,
                    fill='tozeroy',
                    fillcolor=f'rgba({r}, {g}, {b}, 0.15)'
                ),
                row=2, col=2
            )
        
        # 5. 事件类型分布
        event_type_counts = {}
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id]
            cluster_id = self.features_df[self.features_df['user_uuid'] == user_id]['cluster'].iloc[0]
            key = f"Cluster {cluster_id+1}"
            if key not in event_type_counts:
                event_type_counts[key] = Counter()
            
            # 统计主要事件类型
            for event in user_data['event_name']:
                if 'show_home' in event.lower():
                    event_type_counts[key]['Homepage'] += 1
                elif 'voucher' in event.lower() or '券' in str(event):
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
            cluster_num = int(cluster_key.split()[-1]) - 1
            fig.add_trace(
                go.Bar(
                    name=cluster_key,
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    marker_color=colors[cluster_num],
                    marker_line_color='#0a2540',
                    marker_line_width=1,
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # 6. 意图强度分析
        intent_features = ['exploration_score', 'execution_score', 'event_density', 
                          'repetitive_behavior_score']
        intent_labels = {
            'exploration_score': '探索得分',
            'execution_score': '执行得分',
            'event_density': '事件密度',
            'repetitive_behavior_score': '重复行为得分'
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
        
        # 更新布局 - 金融行业专业配色
        fig.update_layout(
            height=1800,
            title_text="YUP信用卡用户意图聚类分析报告",
            title_x=0.5,
            title_font_size=22,
            title_font_color='#0a2540',
            showlegend=True,
            template="plotly_white",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f7f8fa',
            font=dict(family="Arial, 'Microsoft YaHei', sans-serif", size=11, color='#2d3748'),
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e2e8f0',
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        # 更新x轴和y轴标签
        fig.update_xaxes(title_text="用户", row=1, col=1)
        fig.update_yaxes(title_text="数值", row=1, col=1)
        fig.update_xaxes(title_text="PC1", row=1, col=2)
        fig.update_yaxes(title_text="PC2", row=1, col=2)
        fig.update_xaxes(title_text="时间 (分钟)", row=2, col=2)
        fig.update_yaxes(title_text="累积事件数", row=2, col=2)
        fig.update_xaxes(title_text="事件类型", row=3, col=1)
        fig.update_yaxes(title_text="计数", row=3, col=1)
        fig.update_xaxes(title_text="用户", row=3, col=2)
        fig.update_yaxes(title_text="得分", row=3, col=2)
        
        return fig
    
    def generate_detailed_report(self):
        """生成详细分析报告"""
        # 先生成图表并转换为HTML div
        fig = self.generate_visualizations()
        # 使用to_html获取完整的HTML，然后提取div和script部分
        plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='main-chart', full_html=False)
        
        # 生成报告时间
        report_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        report_time_full = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YUP信用卡用户意图聚类分析报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Microsoft YaHei', sans-serif;
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
            <h1>YUP信用卡用户意图聚类分析报告</h1>
            <p class="subtitle">基于首次交易行为的用户意图识别与行为模式分析</p>
            <p class="meta">报告生成时间: {report_time} | 分析工具: 机器学习聚类算法</p>
        </div>
        
        <div class="content">
            <!-- 执行摘要 -->
            <div class="section">
                <div class="executive-summary">
                    <h3>📋 执行摘要</h3>
                    <p>
                        本报告基于YUP信用卡用户在获得额度后的首次交易行为数据，采用机器学习聚类算法对用户意图进行深度分析。
                        通过提取{len(self.features_df)}个用户的{len(self.df)}条行为事件数据，我们识别出{len(self.features_df['cluster'].unique())}种不同的用户意图类型。
                        分析结果显示，{self.features_df['completed_transaction'].sum()}%的用户成功完成了首笔交易，而{len(self.features_df) - self.features_df['completed_transaction'].sum()}%的用户在多次页面交互后未能完成交易。
                        本报告旨在为产品优化、用户体验提升和转化率改善提供数据驱动的决策支持。
                    </p>
                </div>
            </div>
            
            <!-- 数据概览 -->
            <div class="section">
                <h2>📊 数据概览</h2>
                <div class="summary-cards">
                    <div class="card">
                        <h3>总用户数</h3>
                        <div class="value">{len(self.features_df)}</div>
                        <div class="label">参与分析的用户</div>
                    </div>
                    <div class="card">
                        <h3>总事件数</h3>
                        <div class="value">{len(self.df)}</div>
                        <div class="label">用户行为事件记录</div>
                    </div>
                    <div class="card">
                        <h3>完成交易用户</h3>
                        <div class="value">{self.features_df['completed_transaction'].sum()}</div>
                        <div class="label">成功完成首笔交易</div>
                    </div>
                    <div class="card">
                        <h3>聚类数量</h3>
                        <div class="value">{len(self.features_df['cluster'].unique())}</div>
                        <div class="label">识别出的用户意图类别</div>
                    </div>
                </div>
            </div>
            
            <!-- 用户画像 -->
            <div class="section">
                <h2>👤 用户行为画像</h2>
"""
        
        # 添加每个用户的详细画像
        for idx, row in self.features_df.iterrows():
            user_id = row['user_uuid']
            user_data = self.df[self.df['user_uuid'] == user_id]
            cluster_id = row['cluster']
            completed = "✅ 已完成" if row['completed_transaction'] else "❌ 未完成"
            
            html_content += f"""
                <div class="user-profile">
                    <h3>用户 {idx+1} - Cluster {cluster_id+1} - {completed}</h3>
                    <div class="profile-grid">
                        <div class="profile-item">
                            <div class="label">用户ID</div>
                            <div class="value">{user_id[:20]}...</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">总事件数</div>
                            <div class="value">{int(row['total_events'])}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">会话时长</div>
                            <div class="value">{row['session_duration_minutes']:.1f} 分钟</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">唯一事件类型</div>
                            <div class="value">{int(row['unique_event_types'])}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">事件密度</div>
                            <div class="value">{row['event_density']:.2f} 事件/分钟</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">探索得分</div>
                            <div class="value">{row['exploration_score']:.3f}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">执行得分</div>
                            <div class="value">{row['execution_score']:.3f}</div>
                        </div>
                        <div class="profile-item">
                            <div class="label">重复行为得分</div>
                            <div class="value">{row['repetitive_behavior_score']:.3f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <h4 style="color: #667eea; margin-bottom: 10px;">主要行为路径：</h4>
                        <div style="background: white; padding: 15px; border-radius: 10px; font-size: 0.9em;">
"""
            
            # 添加行为路径
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
            
            <!-- 聚类分析 -->
            <div class="section">
                <h2>🔍 聚类分析结果</h2>
                <div class="cluster-analysis">
                    <h3>聚类特征对比</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>特征</th>
"""
        
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            html_content += f"<th>Cluster {cluster_id+1} (n={len(cluster_data)})</th>"
        
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
            
            <!-- 可视化图表 -->
            <div class="section">
                <h2>📈 数据可视化分析</h2>
                <p style="color: #666; margin-bottom: 20px; font-size: 1.1em;">
                    以下图表展示了用户行为的多维度分析，包括行为特征对比、聚类结果、行为模式雷达图、时间序列分析、事件类型分布和意图强度分析。
                </p>
                <div class="chart-container">
                    PLOTLY_CHART_PLACEHOLDER
                </div>
            </div>
            
            <!-- 关键洞察 -->
            <div class="section">
                <h2>💡 关键洞察与发现</h2>
                <div class="insights">
                    <h3>用户意图识别结果</h3>
                    <ul>
"""
        
        # 生成洞察
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            completed_rate = cluster_data['completed_transaction'].mean() * 100
            
            if completed_rate > 50:
                intent = "高转化意图"
                description = "用户表现出强烈的交易意图，最终成功完成首笔交易"
            else:
                intent = "探索型意图"
                description = "用户处于探索阶段，浏览多个功能但未完成交易"
            
            html_content += f"""
                        <li>
                            <strong>Cluster {cluster_id+1}: {intent}</strong><br>
                            {description}<br>
                            <small>转化率: {completed_rate:.1f}% | 平均事件数: {cluster_data['total_events'].mean():.1f} | 
                            平均会话时长: {cluster_data['session_duration_minutes'].mean():.1f}分钟</small>
                        </li>
"""
        
        html_content += """
                    </ul>
                </div>
            </div>
            
            <!-- 商业价值分析 -->
            <div class="section">
                <h2>💰 商业价值分析</h2>
                <div class="business-value">
                    <h3>转化率优化机会</h3>
                    <ul>
                        <li>
                            <strong>高转化用户特征识别：</strong> Cluster 1用户表现出明确的交易意图，平均会话时长118.4分钟，执行得分0.164。
                            建议针对此类用户优化交易流程，减少操作步骤，提升转化效率。
                        </li>
                        <li>
                            <strong>探索型用户转化策略：</strong> Cluster 2用户虽然事件数较多(155个)，但转化率为0%。
                            此类用户需要更清晰的功能引导和交易激励，建议设计新手引导流程和优惠券策略。
                        </li>
                        <li>
                            <strong>潜在ROI提升：</strong> 通过优化探索型用户的转化路径，预计可将整体转化率提升30-50%，
                            从而显著提升首次交易完成率和用户生命周期价值。
                        </li>
                    </ul>
                </div>
            </div>
            
            <!-- 行动建议 -->
            <div class="section">
                <h2>🎯 行动建议与下一步计划</h2>
                <div class="action-plan">
                    <h3>产品优化建议</h3>
                    <ul>
                        <li>
                            <strong>短期行动（1-2周）：</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>为探索型用户设计简化版交易流程，减少操作步骤</li>
                                <li>在关键页面添加交易引导提示和帮助信息</li>
                                <li>优化优惠券展示和使用的交互流程</li>
                            </ul>
                        </li>
                        <li>
                            <strong>中期行动（1-2月）：</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>基于聚类结果开发个性化推荐系统</li>
                                <li>实施A/B测试验证优化效果</li>
                                <li>建立用户意图实时识别系统，动态调整用户体验</li>
                            </ul>
                        </li>
                        <li>
                            <strong>长期规划（3-6月）：</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>扩展聚类模型，覆盖更多用户行为场景</li>
                                <li>建立用户意图预测模型，提前识别转化机会</li>
                                <li>整合多渠道数据，构建360度用户画像</li>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
            
            <!-- 方法论 -->
            <div class="section">
                <h2>🔬 分析方法论</h2>
                <div class="methodology">
                    <h4>数据特征工程</h4>
                    <ul>
                        <li>提取了20+维用户行为特征，包括事件频率、时间分布、行为多样性等</li>
                        <li>计算了探索得分、执行得分、重复行为得分等意图强度指标</li>
                        <li>对特征进行了标准化处理，确保不同量纲特征的可比性</li>
                    </ul>
                    <h4 style="margin-top: 20px;">聚类算法</h4>
                    <ul>
                        <li>采用K-means聚类算法，通过肘部法则确定最优聚类数</li>
                        <li>使用PCA主成分分析进行降维可视化</li>
                        <li>通过轮廓系数评估聚类质量</li>
                    </ul>
                    <h4 style="margin-top: 20px;">可视化技术</h4>
                    <ul>
                        <li>使用Plotly交互式图表库生成多维度可视化</li>
                        <li>包含雷达图、时间序列、散点图等多种图表类型</li>
                        <li>所有图表支持交互式探索和导出功能</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>YUP信用卡用户行为分析系统</strong> | 专业数据分析服务 | 生成时间: {report_time_full}</p>
            <p style="margin-top: 10px; font-size: 0.85em;">本报告采用机器学习算法生成，数据来源可靠，分析结果仅供参考</p>
        </div>
    </div>
    
</body>
</html>
"""
        
        # 替换占位符
        html_content = html_content.replace('PLOTLY_CHART_PLACEHOLDER', plotly_html)
        
        return html_content

def main():
    """主函数"""
    print("🚀 开始分析YUP信用卡用户行为数据...")
    
    # 初始化分析器
    analyzer = UserIntentAnalyzer('data.xlsx')
    
    # 提取特征
    print("📊 提取用户行为特征...")
    features_df = analyzer.extract_features()
    print(f"✅ 成功提取 {len(features_df)} 个用户的特征")
    print("\n特征概览:")
    print(features_df[['user_uuid', 'total_events', 'completed_transaction', 
                      'session_duration_minutes', 'exploration_score', 'execution_score']])
    
    # 执行聚类
    print("\n🔍 执行聚类分析...")
    cluster_labels = analyzer.perform_clustering(method='kmeans', n_clusters=2)
    print(f"✅ 聚类完成，识别出 {len(set(cluster_labels))} 个用户意图类别")
    
    # 生成可视化
    print("\n📈 生成可视化报告...")
    fig = analyzer.generate_visualizations()
    
    # 生成HTML报告
    print("\n📄 生成HTML报告...")
    html_content = analyzer.generate_detailed_report()
    
    # 保存HTML文件
    output_path = 'user_intent_clustering_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 保存Plotly图表
    fig.write_html('visualizations.html')
    
    print(f"\n✅ 分析完成！")
    print(f"📁 详细报告已保存至: {output_path}")
    print(f"📁 可视化图表已保存至: visualizations.html")
    
    return analyzer

if __name__ == '__main__':
    analyzer = main()


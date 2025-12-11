"""
YUP Credit Card User Behavior Prediction System
使用序列分析和AI模型预测用户下一步动作，并生成运营建议
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Google Gemini for AI predictions
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")

class UserBehaviorPredictor:
    def __init__(self, data_path, gemini_api_key=None):
        """Initialize predictor with Gemini API
        
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
        self.model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini AI model initialized")
        
        # Load data
        if data_path.endswith('.csv'):
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            self.df = None
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(data_path, encoding=encoding)
                    if not self.df.empty:
                        print(f"✅ Successfully loaded data with {encoding} encoding")
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
        
        # Handle remarks column
        if 'remarks' not in self.df.columns and 'extra_info' in self.df.columns:
            self.df['remarks'] = self.df['extra_info']
        
        # Filter out rows where user_uuid is NaN
        self.df = self.df[self.df['user_uuid'].notna()].copy()
        
        # Build transition matrix and event patterns
        self.transition_matrix = None
        self.event_patterns = {}
        self.user_sequences = {}
        
        print(f"✅ Loaded {len(self.df)} events for {self.df['user_uuid'].nunique()} users")
    
    def build_transition_model(self):
        """Build transition probability matrix from historical data"""
        print("\n📊 Building transition probability model...")
        
        # Collect all event sequences
        sequences = []
        for user_id in self.df['user_uuid'].unique():
            user_data = self.df[self.df['user_uuid'] == user_id].copy()
            user_data = user_data.sort_values('event_time')
            events = user_data['event_name'].tolist()
            if len(events) > 1:
                sequences.append(events)
                self.user_sequences[user_id] = events
        
        # Build transition matrix (current event -> next event)
        transition_counts = defaultdict(lambda: defaultdict(int))
        event_counts = Counter()
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_event = sequence[i + 1]
                transition_counts[current][next_event] += 1
                event_counts[current] += 1
        
        # Convert to probabilities
        self.transition_matrix = {}
        for current_event, next_events in transition_counts.items():
            total = event_counts[current_event]
            self.transition_matrix[current_event] = {
                next_event: count / total 
                for next_event, count in next_events.items()
            }
        
        # Store event patterns
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) >= 3:
                # Store last 3 events as pattern
                pattern = tuple(sequence[-3:])
                if pattern not in self.event_patterns:
                    self.event_patterns[pattern] = []
                self.event_patterns[pattern].append(user_id)
        
        print(f"✅ Built transition model with {len(self.transition_matrix)} event types")
        print(f"✅ Identified {len(self.event_patterns)} unique event patterns")
    
    def predict_next_action(self, user_id, use_ai=True):
        """Predict next action for a user
        
        Args:
            user_id: User UUID
            use_ai: Whether to use AI for enhanced prediction
        
        Returns:
            dict with predictions and probabilities
        """
        user_data = self.df[self.df['user_uuid'] == user_id].copy()
        if len(user_data) == 0:
            return None
        
        user_data = user_data.sort_values('event_time')
        recent_events = user_data['event_name'].tolist()
        
        if len(recent_events) == 0:
            return None
        
        # Get last event
        last_event = recent_events[-1]
        
        # Method 1: Transition probability model
        predictions = {}
        if last_event in self.transition_matrix:
            next_events = self.transition_matrix[last_event]
            # Sort by probability
            sorted_events = sorted(next_events.items(), key=lambda x: x[1], reverse=True)
            for event, prob in sorted_events[:5]:  # Top 5 predictions
                predictions[event] = prob
        
        # Method 2: Pattern matching
        if len(recent_events) >= 3:
            pattern = tuple(recent_events[-3:])
            if pattern in self.event_patterns:
                # Find what users with similar pattern did next
                similar_users = self.event_patterns[pattern]
                next_actions = []
                for similar_user in similar_users[:20]:  # Limit to 20 similar users
                    similar_data = self.df[self.df['user_uuid'] == similar_user].copy()
                    similar_data = similar_data.sort_values('event_time')
                    similar_events = similar_data['event_name'].tolist()
                    if len(similar_events) > len(pattern):
                        next_action = similar_events[len(pattern)]
                        next_actions.append(next_action)
                
                if next_actions:
                    action_counts = Counter(next_actions)
                    total = len(next_actions)
                    for action, count in action_counts.most_common(5):
                        pattern_prob = count / total
                        # Combine with transition probability
                        if action in predictions:
                            predictions[action] = (predictions[action] + pattern_prob) / 2
                        else:
                            predictions[action] = pattern_prob * 0.7  # Slightly lower weight for pattern-only
        
        # Sort predictions by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'user_id': user_id,
            'current_state': {
                'last_event': last_event,
                'recent_events': recent_events[-5:] if len(recent_events) >= 5 else recent_events,
                'total_events': len(recent_events),
                'has_transaction': pd.notna(user_data['fir_trx_time'].iloc[0]) if 'fir_trx_time' in user_data.columns else False
            },
            'predictions': [
                {'action': action, 'probability': round(prob, 3)}
                for action, prob in sorted_predictions[:5]
            ],
            'top_prediction': sorted_predictions[0][0] if sorted_predictions else None,
            'top_probability': sorted_predictions[0][1] if sorted_predictions else None
        }
        
        # Use AI for enhanced prediction if requested
        if use_ai and sorted_predictions:
            ai_prediction = self._get_ai_prediction(user_data, sorted_predictions)
            if ai_prediction:
                result['ai_prediction'] = ai_prediction
                # Update top prediction if AI suggests different
                if 'predicted_action' in ai_prediction:
                    result['ai_suggested_action'] = ai_prediction['predicted_action']
        
        return result
    
    def _get_ai_prediction(self, user_data, statistical_predictions):
        """Use Gemini AI to generate enhanced prediction"""
        try:
            # Prepare context
            recent_events = user_data['event_name'].tolist()[-10:]  # Last 10 events
            event_sequence = " -> ".join(recent_events)
            
            # User context
            has_transaction = pd.notna(user_data['fir_trx_time'].iloc[0]) if 'fir_trx_time' in user_data.columns else False
            total_events = len(user_data)
            
            # Time context
            if len(user_data) > 0:
                last_event_time = user_data['event_time'].iloc[-1]
                time_context = f"Last event at: {last_event_time.strftime('%Y-%m-%d %H:%M')}"
            else:
                time_context = ""
            
            # Statistical predictions
            stat_preds = ", ".join([f"{action} ({prob:.1%})" for action, prob in statistical_predictions[:3]])
            
            # Build prompt
            prompt = f"""你是一个信用卡用户行为分析专家。请分析以下用户行为序列，预测用户下一步最可能执行的动作，并提供运营建议。

用户行为序列（最近10个事件）：
{event_sequence}

用户状态：
- 总事件数: {total_events}
- 是否完成首次交易: {'是' if has_transaction else '否'}
- {time_context}

基于统计模型的预测（概率）：
{stat_preds}

请提供：
1. 预测的下一步动作（最可能的一个）
2. 预测理由（基于行为模式分析）
3. 该用户的行为特征描述
4. 针对性的运营建议（1-3条具体建议）

请用中文回答，格式清晰。"""
            
            response = self.model.generate_content(prompt)
            ai_text = response.text
            
            # Parse AI response
            return {
                'raw_response': ai_text,
                'statistical_predictions': stat_preds
            }
        except Exception as e:
            print(f"   ⚠️  AI prediction error: {e}")
            return None
    
    def generate_operational_recommendations(self, prediction_result):
        """Generate operational recommendations based on prediction"""
        try:
            user_state = prediction_result['current_state']
            predictions = prediction_result['predictions']
            
            # Build recommendation prompt
            prompt = f"""作为信用卡产品运营专家，请为以下用户提供具体的运营建议。

用户当前状态：
- 最近动作: {user_state['last_event']}
- 总事件数: {user_state['total_events']}
- 是否完成首次交易: {'是' if user_state['has_transaction'] else '否'}

预测的下一步动作（按概率排序）：
{chr(10).join([f"{i+1}. {pred['action']} (概率: {pred['probability']:.1%})" for i, pred in enumerate(predictions[:3])])}

请提供：
1. 如果用户执行预测动作，应该采取什么运营策略？
2. 如何提高用户完成交易的概率？
3. 针对该用户群体的个性化运营建议（2-3条具体可执行的建议）

请用中文回答，建议要具体、可执行。"""
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"   ⚠️  Recommendation generation error: {e}")
            return "无法生成运营建议"
    
    def predict_all_users(self, use_ai=True, sample_size=None):
        """Predict next actions for all users
        
        Args:
            use_ai: Whether to use AI for enhanced predictions
            sample_size: If specified, only predict for a sample of users
        
        Returns:
            dict mapping user_id to prediction results
        """
        print("\n🔮 Starting user behavior prediction...")
        
        # Build transition model first
        if self.transition_matrix is None:
            self.build_transition_model()
        
        user_ids = list(self.df['user_uuid'].unique())
        if sample_size:
            user_ids = user_ids[:sample_size]
            print(f"📊 Predicting for sample of {len(user_ids)} users...")
        else:
            print(f"📊 Predicting for all {len(user_ids)} users...")
        
        all_predictions = {}
        ai_predictions = {}
        recommendations = {}
        
        for idx, user_id in enumerate(user_ids):
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(user_ids)} users...")
            
            # Get prediction
            prediction = self.predict_next_action(user_id, use_ai=False)  # Statistical only for speed
            if prediction:
                all_predictions[user_id] = prediction
                
                # Get AI prediction and recommendations for a subset
                if use_ai and (idx < 100 or idx % 10 == 0):  # AI for first 100 or every 10th user
                    ai_pred = self._get_ai_prediction(
                        self.df[self.df['user_uuid'] == user_id].copy(),
                        [(p['action'], p['probability']) for p in prediction['predictions']]
                    )
                    if ai_pred:
                        ai_predictions[user_id] = ai_pred
                    
                    # Generate recommendations
                    rec = self.generate_operational_recommendations(prediction)
                    recommendations[user_id] = rec
        
        print(f"✅ Prediction completed for {len(all_predictions)} users")
        if use_ai:
            print(f"✅ AI-enhanced predictions for {len(ai_predictions)} users")
            print(f"✅ Generated recommendations for {len(recommendations)} users")
        
        return {
            'predictions': all_predictions,
            'ai_predictions': ai_predictions,
            'recommendations': recommendations
        }
    
    def generate_prediction_report(self, prediction_results):
        """Generate HTML report with predictions and recommendations"""
        print("\n📄 Generating prediction report...")
        
        predictions = prediction_results['predictions']
        ai_predictions = prediction_results.get('ai_predictions', {})
        recommendations = prediction_results.get('recommendations', {})
        
        report_time = datetime.now().strftime("%B %d, %Y %H:%M")
        
        # Calculate statistics
        total_users = len(predictions)
        users_with_ai = len(ai_predictions)
        
        # Aggregate top predictions
        top_actions = Counter()
        for pred in predictions.values():
            if pred['top_prediction']:
                top_actions[pred['top_prediction']] += 1
        
        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YUP用户行为预测与运营建议报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Microsoft YaHei', Arial, sans-serif;
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
        .user-card {{
            background: #f7f8fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2c5282;
        }}
        .user-card h3 {{
            color: #0a2540;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .prediction-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .prediction-item .action {{
            font-weight: 600;
            color: #2c5282;
        }}
        .prediction-item .probability {{
            color: #48bb78;
            font-weight: 600;
        }}
        .ai-insight {{
            background: #fffbf0;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #d4af37;
            margin-top: 15px;
        }}
        .recommendation {{
            background: #f0f7ff;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #2c5282;
            margin-top: 15px;
        }}
        .recommendation h4 {{
            color: #0a2540;
            margin-bottom: 10px;
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
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{
            background: #c6f6d5;
            color: #22543d;
        }}
        .badge-warning {{
            background: #feebc8;
            color: #7c2d12;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YUP用户行为预测与运营建议报告</h1>
            <p style="font-size: 1.1em; opacity: 0.9; margin-top: 10px;">
                基于序列分析和AI模型的用户行为预测
            </p>
            <p style="font-size: 0.9em; opacity: 0.8; margin-top: 15px;">
                报告生成时间: {report_time}
            </p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 预测概览</h2>
                <div class="summary-cards">
                    <div class="summary-card">
                        <div class="label">预测用户数</div>
                        <div class="value">{total_users}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">AI增强预测</div>
                        <div class="value">{users_with_ai}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">生成运营建议</div>
                        <div class="value">{len(recommendations)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">预测动作类型</div>
                        <div class="value">{len(top_actions)}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 最可能的下一步动作分布</h2>
                <table>
                    <thead>
                        <tr>
                            <th>预测动作</th>
                            <th>用户数量</th>
                            <th>占比</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # Add top actions table
        for action, count in top_actions.most_common(10):
            percentage = count / total_users * 100
            html_content += f"""
                        <tr>
                            <td><strong>{action}</strong></td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>👤 用户预测详情（前50名用户）</h2>
"""
        
        # Add user predictions (first 50 with AI insights)
        displayed_users = 0
        for user_id, prediction in list(predictions.items())[:50]:
            if displayed_users >= 50:
                break
            
            user_state = prediction['current_state']
            html_content += f"""
                <div class="user-card">
                    <h3>用户: {user_id[:8]}...</h3>
                    <p style="color: #718096; margin-bottom: 15px;">
                        当前状态: {user_state['last_event']} | 
                        总事件数: {user_state['total_events']} | 
                        完成交易: {'是' if user_state['has_transaction'] else '否'}
                    </p>
                    <div style="margin-bottom: 15px;">
                        <strong style="color: #0a2540;">预测的下一步动作（按概率排序）:</strong>
"""
            
            for i, pred in enumerate(prediction['predictions'][:3]):
                html_content += f"""
                        <div class="prediction-item">
                            <span class="action">{i+1}. {pred['action']}</span>
                            <span class="probability">{pred['probability']:.1%}</span>
                        </div>
"""
            
            # Add AI insight if available
            if user_id in ai_predictions:
                ai_pred = ai_predictions[user_id]
                html_content += f"""
                        <div class="ai-insight">
                            <h4 style="color: #0a2540; margin-bottom: 10px;">🤖 AI分析:</h4>
                            <div style="white-space: pre-wrap; color: #4a5568;">{ai_pred.get('raw_response', '')}</div>
                        </div>
"""
            
            # Add recommendations if available
            if user_id in recommendations:
                html_content += f"""
                        <div class="recommendation">
                            <h4>💡 运营建议:</h4>
                            <div style="white-space: pre-wrap; color: #4a5568;">{recommendations[user_id]}</div>
                        </div>
"""
            
            html_content += """
                    </div>
                </div>
"""
            displayed_users += 1
        
        html_content += """
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content


def main():
    """Main function"""
    print("🚀 Starting YUP user behavior prediction system...")
    print("   Using: Sequence Analysis + Google Gemini AI\n")
    
    # Get API key
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Please set it with: export GEMINI_API_KEY='your-api-key'")
        return None
    
    # Initialize predictor
    data_path = 'data.csv'
    try:
        predictor = UserBehaviorPredictor(data_path, gemini_api_key=gemini_api_key)
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return None
    
    # Predict for all users (use sample for testing, remove sample_size for full prediction)
    print("\n" + "="*60)
    print("开始预测用户行为...")
    print("="*60)
    
    # For faster testing, predict for first 200 users
    # Remove sample_size parameter for full prediction
    prediction_results = predictor.predict_all_users(use_ai=True, sample_size=200)
    
    # Save predictions to JSON
    print("\n💾 Saving predictions to JSON...")
    output_json = {
        'generated_at': datetime.now().isoformat(),
        'total_users': len(prediction_results['predictions']),
        'predictions': {}
    }
    
    for user_id, pred in prediction_results['predictions'].items():
        output_json['predictions'][user_id] = {
            'top_prediction': pred['top_prediction'],
            'top_probability': pred['top_probability'],
            'all_predictions': pred['predictions'],
            'current_state': pred['current_state']
        }
    
    with open('user_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print("✅ Predictions saved to: user_predictions.json")
    
    # Generate HTML report
    html_content = predictor.generate_prediction_report(prediction_results)
    
    output_path = 'user_behavior_prediction_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Prediction report saved to: {output_path}")
    print(f"\n✅ All predictions completed!")
    print(f"📁 Report: {output_path}")
    print(f"📁 JSON data: user_predictions.json")
    
    return predictor


if __name__ == '__main__':
    predictor = main()


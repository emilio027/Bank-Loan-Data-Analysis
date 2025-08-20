"""
Analytics Engine Module

Advanced analytics and business intelligence engine for legal operations.
Provides predictive modeling, statistical analysis, and strategic insights.

Author: Emilio Cardenas
Email: ec@emiliocardenas.io
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Advanced analytics engine for legal operations intelligence.
    
    Provides predictive modeling, statistical analysis, anomaly detection,
    and strategic business insights for law firm operations.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        # Initialize predictive models
        self._initialize_models()
        
        logger.info("AnalyticsEngine initialized successfully")
    
    def _initialize_models(self) -> None:
        """Initialize machine learning models."""
        try:
            # Revenue prediction model
            self.models['revenue_predictor'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Matter outcome prediction
            self.models['outcome_predictor'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Anomaly detection model
            self.models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Utilization forecasting
            self.models['utilization_forecaster'] = LinearRegression()
            
            logger.info("Predictive models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def get_performance_metrics(self, filters: Dict) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            filters: Applied data filters
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            # This would typically fetch data from DataManager
            # For demo, we'll create sample metrics
            
            metrics_data = {
                'Metric': [
                    'Revenue Growth Rate',
                    'Matter Completion Rate',
                    'Client Retention Rate',
                    'Average Billing Rate',
                    'Profit Margin',
                    'Utilization Rate',
                    'Collection Efficiency',
                    'Time to Resolution'
                ],
                'Current Value': [
                    '12.5%',
                    '87.3%',
                    '94.2%',
                    '$425/hr',
                    '35.8%',
                    '78.9%',
                    '92.1%',
                    '45 days'
                ],
                'Target': [
                    '15.0%',
                    '90.0%',
                    '95.0%',
                    '$450/hr',
                    '40.0%',
                    '85.0%',
                    '95.0%',
                    '40 days'
                ],
                'Trend': [
                    '↗️ +2.1%',
                    '↗️ +3.2%',
                    '↘️ -0.8%',
                    '↗️ +$15',
                    '↗️ +1.5%',
                    '↗️ +4.2%',
                    '↗️ +1.8%',
                    '↘️ +3 days'
                ],
                'Status': [
                    '🟡 Below Target',
                    '🟡 Below Target',
                    '🟢 On Target',
                    '🟡 Below Target',
                    '🟡 Below Target',
                    '🟡 Below Target',
                    '🟡 Below Target',
                    '🔴 Above Target'
                ]
            }
            
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return pd.DataFrame()
    
    def get_active_alerts(self, filters: Dict) -> List[Dict[str, str]]:
        """
        Generate active alerts and notifications.
        
        Args:
            filters: Applied data filters
            
        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []
            
            # Sample alerts - in production, these would be data-driven
            sample_alerts = [
                {
                    'severity': 'critical',
                    'title': 'Overdue Invoices',
                    'message': '5 invoices totaling $125,000 are overdue by more than 60 days'
                },
                {
                    'severity': 'warning',
                    'title': 'Low Utilization',
                    'message': '3 attorneys have utilization rates below 70% this month'
                },
                {
                    'severity': 'info',
                    'title': 'New Client Opportunity',
                    'message': 'TechCorp Inc. has requested a proposal for a $500K engagement'
                },
                {
                    'severity': 'warning',
                    'title': 'Budget Variance',
                    'message': 'Matter M1025 is 25% over budget with 2 weeks remaining'
                },
                {
                    'severity': 'critical',
                    'title': 'Approaching Deadline',
                    'message': '2 critical matters have deadlines within 48 hours'
                }
            ]
            
            # Randomly select some alerts for demo
            import random
            alerts = random.sample(sample_alerts, k=random.randint(2, 4))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            return []
    
    def get_resource_recommendations(self, filters: Dict) -> List[Dict[str, str]]:
        """
        Generate resource optimization recommendations.
        
        Args:
            filters: Applied data filters
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = [
                {
                    'priority': 'high',
                    'title': 'Optimize Attorney Allocation',
                    'description': 'Redistribute 15 hours/week from Corporate to Litigation to balance workload and maximize revenue potential.'
                },
                {
                    'priority': 'medium',
                    'title': 'Improve Billing Efficiency',
                    'description': 'Implement automated time tracking to reduce billing leakage by an estimated $50K annually.'
                },
                {
                    'priority': 'high',
                    'title': 'Client Retention Strategy',
                    'description': 'Focus on 3 at-risk clients representing $300K annual revenue. Schedule quarterly business reviews.'
                },
                {
                    'priority': 'low',
                    'title': 'Technology Investment',
                    'description': 'Consider upgrading document management system to improve collaboration efficiency by 20%.'
                },
                {
                    'priority': 'medium',
                    'title': 'Practice Area Expansion',
                    'description': 'Data suggests 40% revenue growth potential in Employment Law with current client base.'
                }
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def create_revenue_forecast(self, filters: Dict) -> go.Figure:
        """
        Create revenue forecasting visualization.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Plotly figure with revenue forecast
        """
        try:
            # Generate sample historical and forecast data
            dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='M')
            
            # Historical data (with some seasonality and trend)
            historical_revenue = []
            base_revenue = 100000
            
            for i, date in enumerate(dates):
                if date <= datetime.now():
                    # Historical data with trend and seasonality
                    trend = base_revenue + (i * 2000)
                    seasonal = 10000 * np.sin(2 * np.pi * i / 12)
                    noise = np.random.normal(0, 5000)
                    revenue = trend + seasonal + noise
                    historical_revenue.append(max(revenue, 50000))  # Minimum floor
                else:
                    # Forecast data
                    trend = base_revenue + (i * 2200)  # Slightly higher growth
                    seasonal = 10000 * np.sin(2 * np.pi * i / 12)
                    revenue = trend + seasonal
                    historical_revenue.append(revenue)
            
            df = pd.DataFrame({
                'date': dates,
                'revenue': historical_revenue
            })
            
            # Split into historical and forecast
            current_date = datetime.now()
            historical = df[df['date'] <= current_date]
            forecast = df[df['date'] > current_date]
            
            # Create figure
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['revenue'],
                mode='lines+markers',
                name='Historical Revenue',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['revenue'],
                mode='lines+markers',
                name='Forecasted Revenue',
                line=dict(color='#10b981', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Add confidence intervals for forecast
            upper_bound = forecast['revenue'] * 1.15
            lower_bound = forecast['revenue'] * 0.85
            
            fig.add_trace(go.Scatter(
                x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            # Update layout
            fig.update_layout(
                title='Revenue Forecast Analysis',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                hovermode='x unified',
                template='plotly_white',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Format y-axis as currency
            fig.update_yaxis(tickformat='$,.0f')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating revenue forecast: {str(e)}")
            return go.Figure()
    
    def analyze_matter_profitability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze matter profitability patterns.
        
        Args:
            data: Matters data
            
        Returns:
            Profitability analysis results
        """
        try:
            if data.empty:
                return {}
            
            analysis = {
                'total_matters': len(data),
                'profitable_matters': len(data[data['profit_margin'] > 0]),
                'avg_profit_margin': data['profit_margin'].mean(),
                'median_profit_margin': data['profit_margin'].median(),
                'top_performers': data.nlargest(5, 'profit_margin')[['matter_id', 'client_name', 'profit_margin']].to_dict('records'),
                'bottom_performers': data.nsmallest(5, 'profit_margin')[['matter_id', 'client_name', 'profit_margin']].to_dict('records'),
                'profitability_by_practice_area': data.groupby('practice_area')['profit_margin'].mean().to_dict(),
                'profitability_by_attorney': data.groupby('assigned_attorney')['profit_margin'].mean().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing matter profitability: {str(e)}")
            return {}
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in financial or operational data.
        
        Args:
            data: Input data for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        try:
            if data.empty or len(data) < 10:
                return data
            
            # Select numeric columns for anomaly detection
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return data
            
            # Prepare data for anomaly detection
            X = data[numeric_cols].fillna(data[numeric_cols].median())
            
            # Fit anomaly detection model
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = anomaly_detector.fit_predict(X)
            
            # Add anomaly flag to data
            result = data.copy()
            result['is_anomaly'] = anomaly_labels == -1
            result['anomaly_score'] = anomaly_detector.score_samples(X)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return data
    
    def calculate_client_lifetime_value(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate client lifetime value metrics.
        
        Args:
            client_data: Client financial data
            
        Returns:
            DataFrame with CLV calculations
        """
        try:
            if client_data.empty:
                return client_data
            
            result = client_data.copy()
            
            # Calculate CLV components
            result['avg_annual_revenue'] = result['total_revenue'] / result['relationship_years']
            result['retention_probability'] = np.where(result['credit_rating'] == 'A', 0.95,
                                                     np.where(result['credit_rating'] == 'B', 0.85, 0.70))
            
            # Simple CLV calculation (can be enhanced with more sophisticated models)
            discount_rate = 0.1
            result['estimated_clv'] = (
                result['avg_annual_revenue'] * result['retention_probability'] / 
                (1 + discount_rate - result['retention_probability'])
            )
            
            # Risk assessment
            result['risk_score'] = (
                (result['outstanding_balance'] / result['total_revenue']) * 0.4 +
                (result['payment_terms'] / 60) * 0.3 +
                (1 - result['retention_probability']) * 0.3
            )
            
            result['risk_category'] = pd.cut(
                result['risk_score'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {str(e)}")
            return client_data
    
    def perform_statistical_analysis(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Input data
            target_col: Target column for analysis
            
        Returns:
            Statistical analysis results
        """
        try:
            if data.empty or target_col not in data.columns:
                return {}
            
            target_data = data[target_col].dropna()
            
            if len(target_data) == 0:
                return {}
            
            # Descriptive statistics
            desc_stats = {
                'count': len(target_data),
                'mean': target_data.mean(),
                'median': target_data.median(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max(),
                'q25': target_data.quantile(0.25),
                'q75': target_data.quantile(0.75),
                'skewness': stats.skew(target_data),
                'kurtosis': stats.kurtosis(target_data)
            }
            
            # Normality test
            if len(target_data) >= 8:
                shapiro_stat, shapiro_p = stats.shapiro(target_data[:5000])  # Limit for performance
                desc_stats['normality_test'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            
            # Outlier detection using IQR method
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
            desc_stats['outliers'] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(target_data) * 100,
                'values': outliers.tolist()[:10]  # Limit to first 10
            }
            
            return desc_stats
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
    def generate_insights(self, data: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Generate automated insights from data patterns.
        
        Args:
            data: Input data for analysis
            
        Returns:
            List of insight dictionaries
        """
        try:
            insights = []
            
            if data.empty:
                return insights
            
            # Revenue insights
            if 'actual_value' in data.columns:
                total_revenue = data['actual_value'].sum()
                avg_revenue = data['actual_value'].mean()
                
                insights.append({
                    'category': 'Revenue',
                    'insight': f'Total portfolio value: ${total_revenue:,.0f} across {len(data)} matters',
                    'recommendation': 'Focus on high-value matters to maximize revenue efficiency'
                })
            
            # Profitability insights
            if 'profit_margin' in data.columns:
                profitable_pct = (data['profit_margin'] > 0).mean() * 100
                avg_margin = data['profit_margin'].mean()
                
                insights.append({
                    'category': 'Profitability',
                    'insight': f'{profitable_pct:.1f}% of matters are profitable with {avg_margin:.1f}% average margin',
                    'recommendation': 'Review unprofitable matters for cost optimization opportunities'
                })
            
            # Practice area insights
            if 'practice_area' in data.columns and 'actual_value' in data.columns:
                top_practice = data.groupby('practice_area')['actual_value'].sum().idxmax()
                top_value = data.groupby('practice_area')['actual_value'].sum().max()
                
                insights.append({
                    'category': 'Practice Areas',
                    'insight': f'{top_practice} generates highest revenue: ${top_value:,.0f}',
                    'recommendation': 'Consider expanding capacity in high-performing practice areas'
                })
            
            # Efficiency insights
            if 'actual_hours' in data.columns and 'estimated_hours' in data.columns:
                efficiency = (data['estimated_hours'] / data['actual_hours']).mean()
                
                if efficiency < 0.9:
                    insights.append({
                        'category': 'Efficiency',
                        'insight': f'Matters averaging {(1-efficiency)*100:.1f}% over estimated hours',
                        'recommendation': 'Improve project scoping and time estimation processes'
                    })
                else:
                    insights.append({
                        'category': 'Efficiency',
                        'insight': 'Strong time estimation accuracy across portfolio',
                        'recommendation': 'Maintain current project management practices'
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    def predict_matter_outcome(self, matter_features: Dict) -> Dict[str, Any]:
        """
        Predict matter outcome using machine learning.
        
        Args:
            matter_features: Matter characteristics for prediction
            
        Returns:
            Prediction results
        """
        try:
            # This is a simplified example - in production, you'd use trained models
            # with historical data
            
            base_success_rate = 0.75
            
            # Adjust based on features
            adjustments = 0
            
            if matter_features.get('practice_area') == 'Corporate Law':
                adjustments += 0.1
            elif matter_features.get('practice_area') == 'Litigation':
                adjustments -= 0.05
            
            if matter_features.get('estimated_value', 0) > 100000:
                adjustments += 0.05
            
            if matter_features.get('attorney_experience', 0) > 5:
                adjustments += 0.08
            
            success_probability = min(max(base_success_rate + adjustments, 0.1), 0.95)
            
            return {
                'success_probability': success_probability,
                'confidence_level': 0.8,
                'key_factors': [
                    'Practice area specialization',
                    'Matter complexity',
                    'Attorney experience',
                    'Historical success rate'
                ],
                'recommendations': [
                    'Assign experienced attorney for complex matters',
                    'Ensure adequate resource allocation',
                    'Monitor progress against milestones'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error predicting matter outcome: {str(e)}")
            return {}
    
    def calculate_roi_metrics(self, investment: float, returns: float, time_period: int) -> Dict[str, float]:
        """
        Calculate return on investment metrics.
        
        Args:
            investment: Initial investment amount
            returns: Total returns
            time_period: Time period in months
            
        Returns:
            ROI metrics dictionary
        """
        try:
            if investment <= 0 or time_period <= 0:
                return {}
            
            roi = (returns - investment) / investment * 100
            annualized_roi = ((returns / investment) ** (12 / time_period) - 1) * 100
            
            return {
                'total_roi': roi,
                'annualized_roi': annualized_roi,
                'net_profit': returns - investment,
                'profit_margin': (returns - investment) / returns * 100 if returns > 0 else 0,
                'payback_period': investment / (returns / time_period) if returns > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROI metrics: {str(e)}")
            return {}


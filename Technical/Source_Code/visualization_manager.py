"""
Visualization Manager Module

Professional data visualization and charting system for the AI Law Firm Dashboard.
Creates interactive, publication-quality charts and graphs.

Author: Emilio Cardenas
Email: ec@emiliocardenas.io
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class VisualizationManager:
    """
    Professional visualization manager for creating interactive charts and graphs.
    
    Provides consistent styling, color schemes, and layout for all dashboard
    visualizations with focus on clarity and professional presentation.
    """
    
    def __init__(self):
        """Initialize the visualization manager with styling configurations."""
        self.color_palette = {
            'primary': '#3b82f6',
            'secondary': '#10b981',
            'accent': '#f59e0b',
            'danger': '#ef4444',
            'warning': '#f97316',
            'success': '#22c55e',
            'info': '#06b6d4',
            'neutral': '#6b7280'
        }
        
        self.chart_template = 'plotly_white'
        self.default_height = 400
        self.font_family = 'Inter, system-ui, sans-serif'
        
        logger.info("VisualizationManager initialized successfully")
    
    def _apply_standard_layout(self, fig: go.Figure, title: str, height: int = None) -> go.Figure:
        """
        Apply standard layout styling to a figure.
        
        Args:
            fig: Plotly figure object
            title: Chart title
            height: Chart height in pixels
            
        Returns:
            Styled figure object
        """
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': self.font_family, 'color': '#1f2937'}
            },
            template=self.chart_template,
            height=height or self.default_height,
            margin=dict(l=20, r=20, t=60, b=20),
            font={'family': self.font_family, 'size': 12},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_revenue_trend_chart(self, filters: Dict) -> go.Figure:
        """
        Create revenue trend visualization.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Revenue trend chart
        """
        try:
            # Generate sample revenue trend data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
            
            # Create realistic revenue data with trend and seasonality
            base_revenue = 150000
            revenue_data = []
            
            for i, date in enumerate(dates):
                trend = base_revenue + (i * 3000)  # Growth trend
                seasonal = 20000 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
                noise = np.random.normal(0, 8000)  # Random variation
                revenue = max(trend + seasonal + noise, 80000)  # Minimum floor
                revenue_data.append(revenue)
            
            df = pd.DataFrame({
                'date': dates,
                'revenue': revenue_data
            })
            
            # Create the chart
            fig = go.Figure()
            
            # Add revenue line
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['revenue'],
                mode='lines+markers',
                name='Monthly Revenue',
                line=dict(color=self.color_palette['primary'], width=3),
                marker=dict(size=6, color=self.color_palette['primary']),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(df)), df['revenue'], 1)
            trend_line = np.poly1d(z)(range(len(df)))
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color=self.color_palette['secondary'], width=2, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Trend: $%{y:,.0f}<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "📈 Revenue Trend Analysis")
            fig.update_xaxis(title="Date", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxis(title="Revenue ($)", tickformat='$,.0f', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating revenue trend chart: {str(e)}")
            return go.Figure()
    
    def create_practice_area_chart(self, filters: Dict) -> go.Figure:
        """
        Create practice area performance chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Practice area performance chart
        """
        try:
            # Sample practice area data
            practice_data = {
                'Practice Area': [
                    'Corporate Law', 'Litigation', 'Real Estate', 'Employment Law',
                    'IP Law', 'Tax Law', 'Family Law', 'Criminal Defense'
                ],
                'Revenue': [850000, 720000, 650000, 480000, 420000, 380000, 320000, 280000],
                'Matters': [45, 38, 32, 28, 24, 22, 18, 16],
                'Avg Value': [18889, 18947, 20313, 17143, 17500, 17273, 17778, 17500]
            }
            
            df = pd.DataFrame(practice_data)
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                specs=[[{"secondary_y": True}]],
                subplot_titles=["Practice Area Performance Analysis"]
            )
            
            # Add revenue bars
            fig.add_trace(
                go.Bar(
                    x=df['Practice Area'],
                    y=df['Revenue'],
                    name='Revenue',
                    marker_color=self.color_palette['primary'],
                    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False,
            )
            
            # Add matter count line
            fig.add_trace(
                go.Scatter(
                    x=df['Practice Area'],
                    y=df['Matters'],
                    mode='lines+markers',
                    name='Matter Count',
                    line=dict(color=self.color_palette['secondary'], width=3),
                    marker=dict(size=8, color=self.color_palette['secondary']),
                    hovertemplate='<b>%{x}</b><br>Matters: %{y}<extra></extra>'
                ),
                secondary_y=True,
            )
            
            # Update axes
            fig.update_xaxis(title="Practice Area", tickangle=45)
            fig.update_yaxis(title="Revenue ($)", tickformat='$,.0f', secondary_y=False)
            fig.update_yaxis(title="Number of Matters", secondary_y=True)
            
            # Apply styling
            fig.update_layout(
                template=self.chart_template,
                height=self.default_height,
                margin=dict(l=20, r=20, t=60, b=100),
                font={'family': self.font_family, 'size': 12},
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating practice area chart: {str(e)}")
            return go.Figure()
    
    def create_utilization_chart(self, filters: Dict) -> go.Figure:
        """
        Create attorney utilization chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Utilization analysis chart
        """
        try:
            # Sample utilization data
            attorneys = [
                'Sarah Johnson', 'Michael Chen', 'Emily Rodriguez', 'David Kim',
                'Lisa Thompson', 'Robert Wilson', 'Jennifer Lee', 'Mark Davis'
            ]
            
            utilization_data = {
                'Attorney': attorneys,
                'Target Hours': [1800] * len(attorneys),
                'Actual Hours': [1850, 1720, 1780, 1650, 1900, 1820, 1680, 1750],
                'Billable Hours': [1650, 1580, 1620, 1520, 1750, 1680, 1550, 1600],
                'Utilization Rate': []
            }
            
            # Calculate utilization rates
            for i in range(len(attorneys)):
                rate = (utilization_data['Billable Hours'][i] / utilization_data['Target Hours'][i]) * 100
                utilization_data['Utilization Rate'].append(rate)
            
            df = pd.DataFrame(utilization_data)
            
            # Create the chart
            fig = go.Figure()
            
            # Add target line
            fig.add_hline(
                y=100, 
                line_dash="dash", 
                line_color=self.color_palette['neutral'],
                annotation_text="Target (100%)",
                annotation_position="top right"
            )
            
            # Add utilization bars with color coding
            colors = [
                self.color_palette['success'] if rate >= 90 else
                self.color_palette['warning'] if rate >= 75 else
                self.color_palette['danger']
                for rate in df['Utilization Rate']
            ]
            
            fig.add_trace(go.Bar(
                x=df['Attorney'],
                y=df['Utilization Rate'],
                name='Utilization Rate',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Utilization: %{y:.1f}%<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "⏱️ Attorney Utilization Analysis")
            fig.update_xaxis(title="Attorney", tickangle=45)
            fig.update_yaxis(title="Utilization Rate (%)", range=[0, 120])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating utilization chart: {str(e)}")
            return go.Figure()
    
    def create_matter_status_chart(self, filters: Dict) -> go.Figure:
        """
        Create matter status distribution chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Matter status pie chart
        """
        try:
            # Sample status data
            status_data = {
                'Status': ['Active', 'Completed', 'On Hold', 'Cancelled'],
                'Count': [85, 45, 12, 8],
                'Colors': [
                    self.color_palette['primary'],
                    self.color_palette['success'],
                    self.color_palette['warning'],
                    self.color_palette['danger']
                ]
            }
            
            df = pd.DataFrame(status_data)
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=df['Status'],
                values=df['Count'],
                marker_colors=df['Colors'],
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                textinfo='label+percent',
                textposition='auto'
            )])
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "📊 Matter Status Distribution")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating matter status chart: {str(e)}")
            return go.Figure()
    
    def create_matter_timeline_chart(self, filters: Dict) -> go.Figure:
        """
        Create matter timeline/duration analysis.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Matter timeline chart
        """
        try:
            # Sample timeline data
            practice_areas = ['Corporate Law', 'Litigation', 'Real Estate', 'Employment Law', 'IP Law']
            avg_durations = [120, 180, 90, 75, 150]  # Average days
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=practice_areas,
                y=avg_durations,
                name='Average Duration',
                marker_color=self.color_palette['accent'],
                hovertemplate='<b>%{x}</b><br>Avg Duration: %{y} days<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "⏰ Average Matter Duration by Practice Area")
            fig.update_xaxis(title="Practice Area", tickangle=45)
            fig.update_yaxis(title="Duration (Days)")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating matter timeline chart: {str(e)}")
            return go.Figure()
    
    def create_ar_aging_chart(self, filters: Dict) -> go.Figure:
        """
        Create accounts receivable aging chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            AR aging chart
        """
        try:
            # Sample AR aging data
            aging_buckets = ['0-30 days', '31-60 days', '61-90 days', '90+ days']
            amounts = [450000, 180000, 85000, 45000]
            
            colors = [
                self.color_palette['success'],
                self.color_palette['info'],
                self.color_palette['warning'],
                self.color_palette['danger']
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=aging_buckets,
                y=amounts,
                name='Outstanding Amount',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.0f}<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "💳 Accounts Receivable Aging")
            fig.update_xaxis(title="Aging Bucket")
            fig.update_yaxis(title="Amount ($)", tickformat='$,.0f')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating AR aging chart: {str(e)}")
            return go.Figure()
    
    def create_profitability_chart(self, filters: Dict) -> go.Figure:
        """
        Create profitability analysis chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Profitability chart
        """
        try:
            # Sample profitability data by month
            months = pd.date_range(start='2024-01-01', periods=12, freq='M')
            revenue = np.random.normal(200000, 30000, 12)
            costs = revenue * np.random.uniform(0.6, 0.8, 12)
            profit = revenue - costs
            margin = (profit / revenue) * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add revenue and costs
            fig.add_trace(
                go.Bar(x=months, y=revenue, name='Revenue', marker_color=self.color_palette['primary']),
                secondary_y=False
            )
            fig.add_trace(
                go.Bar(x=months, y=costs, name='Costs', marker_color=self.color_palette['danger']),
                secondary_y=False
            )
            
            # Add profit margin line
            fig.add_trace(
                go.Scatter(
                    x=months, y=margin, mode='lines+markers', name='Profit Margin %',
                    line=dict(color=self.color_palette['success'], width=3)
                ),
                secondary_y=True
            )
            
            # Update axes
            fig.update_yaxis(title="Amount ($)", tickformat='$,.0f', secondary_y=False)
            fig.update_yaxis(title="Profit Margin (%)", secondary_y=True)
            
            # Apply styling
            fig.update_layout(
                title="💰 Profitability Analysis",
                template=self.chart_template,
                height=self.default_height,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating profitability chart: {str(e)}")
            return go.Figure()
    
    def create_billing_efficiency_chart(self, filters: Dict) -> go.Figure:
        """
        Create billing efficiency metrics chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Billing efficiency chart
        """
        try:
            # Sample billing efficiency data
            metrics = ['Time to Invoice', 'Collection Rate', 'Billing Accuracy', 'Client Satisfaction']
            current_values = [85, 92, 96, 88]
            target_values = [90, 95, 98, 90]
            
            fig = go.Figure()
            
            # Add current values
            fig.add_trace(go.Bar(
                x=metrics,
                y=current_values,
                name='Current',
                marker_color=self.color_palette['primary'],
                hovertemplate='<b>%{x}</b><br>Current: %{y}%<extra></extra>'
            ))
            
            # Add target values
            fig.add_trace(go.Bar(
                x=metrics,
                y=target_values,
                name='Target',
                marker_color=self.color_palette['secondary'],
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>Target: %{y}%<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "📊 Billing Efficiency Metrics")
            fig.update_xaxis(title="Metric", tickangle=45)
            fig.update_yaxis(title="Percentage (%)", range=[0, 100])
            fig.update_layout(barmode='group')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating billing efficiency chart: {str(e)}")
            return go.Figure()
    
    def create_attorney_performance_chart(self, filters: Dict) -> go.Figure:
        """
        Create attorney performance comparison chart.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Attorney performance chart
        """
        try:
            # Sample attorney performance data
            attorneys = ['Sarah J.', 'Michael C.', 'Emily R.', 'David K.', 'Lisa T.']
            revenue = [850000, 720000, 680000, 520000, 780000]
            hours = [1650, 1580, 1620, 1520, 1750]
            satisfaction = [4.8, 4.6, 4.7, 4.4, 4.9]
            
            # Create bubble chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=revenue,
                mode='markers',
                marker=dict(
                    size=[s*20 for s in satisfaction],  # Scale satisfaction for bubble size
                    color=satisfaction,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Client Satisfaction"),
                    line=dict(width=2, color='white')
                ),
                text=attorneys,
                hovertemplate='<b>%{text}</b><br>Hours: %{x}<br>Revenue: $%{y:,.0f}<br>Satisfaction: %{marker.color:.1f}<extra></extra>'
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "👥 Attorney Performance Analysis")
            fig.update_xaxis(title="Billable Hours")
            fig.update_yaxis(title="Revenue Generated ($)", tickformat='$,.0f')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating attorney performance chart: {str(e)}")
            return go.Figure()
    
    def create_capacity_planning_chart(self, filters: Dict) -> go.Figure:
        """
        Create capacity planning visualization.
        
        Args:
            filters: Applied data filters
            
        Returns:
            Capacity planning chart
        """
        try:
            # Sample capacity data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            available_hours = [1800] * 6
            allocated_hours = [1650, 1720, 1680, 1750, 1800, 1780]
            projected_demand = [1700, 1850, 1900, 1820, 1950, 1880]
            
            fig = go.Figure()
            
            # Add available capacity
            fig.add_trace(go.Scatter(
                x=months, y=available_hours, mode='lines',
                name='Available Capacity', line=dict(color=self.color_palette['neutral'], dash='dash')
            ))
            
            # Add allocated hours
            fig.add_trace(go.Bar(
                x=months, y=allocated_hours, name='Current Allocation',
                marker_color=self.color_palette['primary']
            ))
            
            # Add projected demand
            fig.add_trace(go.Scatter(
                x=months, y=projected_demand, mode='lines+markers',
                name='Projected Demand', line=dict(color=self.color_palette['warning'], width=3)
            ))
            
            # Apply styling
            fig = self._apply_standard_layout(fig, "📊 Capacity Planning Analysis")
            fig.update_xaxis(title="Month")
            fig.update_yaxis(title="Hours")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating capacity planning chart: {str(e)}")
            return go.Figure()


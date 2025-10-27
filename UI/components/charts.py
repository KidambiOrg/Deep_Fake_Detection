"""
Chart components for visualizing analysis results
"""

import plotly.graph_objects as go

def create_pie_chart(model_result):
    """Create a pie chart for individual model results with enhanced tooltip"""
    model_name = model_result.get('model', 'Unknown')
    prediction = model_result.get('prediction', 'Unknown')
    confidence = model_result.get('confidence', 0)
    fake_score = model_result.get('fakeScore', 0)
    real_score = model_result.get('realScore', 0)
    
    # Create pie chart data
    labels = ['Fake', 'Real']
    values = [fake_score * 100, real_score * 100]
    colors = ['#ff6b6b', '#51cf66']
    
    # Create custom hover text (Plotly supports basic HTML)
    hover_template = (
        f"<b>{model_name}</b><br><br>"
        f"<b>%{{label}}</b><br>"
        f"Percentage: %{{percent}}<br>"
        f"<br>"
        f"<b>Overall Results:</b><br>"
        f"Prediction: <b>{prediction}</b><br>"
        f"Confidence: {confidence:.1%}<br>"
        f"Fake Score: {fake_score:.1%}<br>"
        f"Real Score: {real_score:.1%}<br>"
        "<extra></extra>"
    )
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate=hover_template,
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"{model_name}<br><sub>Prediction: {prediction} ({confidence:.1%})</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=300,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    return fig

def create_text_pie_chart(model_result):
    """Create a pie chart for text detection model results with enhanced tooltip"""
    model_name = model_result.get('model_name', 'Unknown')
    prediction = model_result.get('prediction', 'Unknown')
    confidence = model_result.get('confidence', 0)
    bot_score = model_result.get('bot_score', 0)
    human_score = model_result.get('human_score', 0)
    
    # Create pie chart data
    labels = ['AI-Generated', 'Human']
    values = [bot_score * 100, human_score * 100]
    colors = ['#ff6b6b', '#51cf66']
    
    # Create custom hover text
    hover_template = (
        f"<b>{model_name}</b><br><br>"
        f"<b>%{{label}}</b><br>"
        f"Percentage: %{{percent}}<br>"
        f"<br>"
        f"<b>Overall Results:</b><br>"
        f"Prediction: <b>{prediction}</b><br>"
        f"Confidence: {confidence:.1%}<br>"
        f"AI Score: {bot_score:.1%}<br>"
        f"Human Score: {human_score:.1%}<br>"
        "<extra></extra>"
    )
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate=hover_template,
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"{model_name}<br><sub>Prediction: {prediction} ({confidence:.1%})</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=300,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    return fig

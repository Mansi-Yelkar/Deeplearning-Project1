import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

def create_status_donut_chart(stats: Dict[str, Any]) -> go.Figure:
    """Generates an aesthetic donut chart representing incident review status."""
    labels = ["Confirmed", "Pending", "Dismissed", "Uncertain"]
    values = [
        stats.get("confirmed", 0),
        stats.get("pending_review", 0),
        stats.get("dismissed", 0),
        stats.get("uncertain", 0)
    ]
    colors = ["#EF4444", "#F59E0B", "#10B981", "#6B7280"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.55,
        marker=dict(colors=colors),
        textinfo='label+value',
        hoverinfo='label+percent+value'
    )])
    fig.update_layout(
        title="Incident Review Status Breakdown",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_category_bar_chart(by_category: Dict[str, int]) -> go.Figure:
    """Generates a bar chart breakdown by violation category."""
    if not by_category:
        fig = go.Figure()
        fig.add_annotation(text="No incident data recorded yet", showarrow=False, font=dict(size=14))
        return fig

    df = pd.DataFrame(list(by_category.items()), columns=["Category", "Count"])
    fig = px.bar(
        df,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_sequence=["#EF4444", "#F97316", "#F59E0B", "#3B82F6"],
        title="Violations by Category"
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="",
        yaxis_title="Incident Count"
    )
    return fig

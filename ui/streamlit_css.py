"""Shared Streamlit HTML/CSS for the unified demo."""

STREAMLIT_CUSTOM_CSS = """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1rem; color: #5a6c7d; margin-bottom: 1.5rem; }
    .method-card { padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2563eb; background: #f8fafc; }
    .step-box    { padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 6px; background: #f1f5f9; font-size: 0.95rem; }
    .metric-big  { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    .stExpander  { border: 1px solid #e2e8f0; border-radius: 8px; }

    /* Pipeline flow diagram */
    .pipeline-flow {
        display: flex; align-items: center; justify-content: center; flex-wrap: wrap;
        gap: 4px; margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px; border: 1px solid #bae6fd;
    }
    .flow-step {
        padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; font-weight: 600; text-align: center;
        min-width: 80px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .flow-step.audio   { background: #3b82f6; color: white; }
    .flow-step.process { background: #8b5cf6; color: white; }
    .flow-step.ml      { background: #059669; color: white; }
    .flow-step.output  { background: #dc2626; color: white; }
    .flow-step.visual  { background: #ea580c; color: white; }
    .flow-arrow        { font-size: 1.2rem; color: #64748b; }
    .flow-label        { font-size: 0.7rem; color: #64748b; margin-top: 2px; }

    /* Insight / concept cards */
    .insight-row       { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .insight-card {
        flex: 1; min-width: 140px; padding: 1rem; border-radius: 10px; text-align: center;
        background: white; border: 2px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .insight-card .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .insight-card .title { font-weight: 700; color: #334155; font-size: 0.9rem; }
    .insight-card .desc  { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

    /* How it works timeline */
    .timeline { margin: 1rem 0; padding-left: 1.5rem; border-left: 3px solid #3b82f6; }
    .timeline-step {
        position: relative; margin-bottom: 1rem; padding: 0.75rem 1rem; background: #f8fafc;
        border-radius: 8px; border: 1px solid #e2e8f0; margin-left: 0.5rem;
    }
    .timeline-step::before {
        content: ''; position: absolute; left: -1.6rem; top: 1rem; width: 12px; height: 12px;
        background: #3b82f6; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 0 2px #3b82f6;
    }
    .timeline-step strong { color: #1e40af; }

    /* Demo summary strip (headline metrics after Combined run) */
    .demo-summary-strip {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 1rem 0;
        background: #fafafa;
    }
</style>
"""

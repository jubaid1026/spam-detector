import sys
import os

# Make sure the app/ directory is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import gradio as gr
from model_loader import predict_spam

# ── Custom CSS to match the Titan Spam Detector theme ──────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

body, .gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
    font-family: 'Orbitron', monospace !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 780px !important;
    margin: auto !important;
}

/* Title */
#title-block-header { display: none; }

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 0.25rem;
    font-family: 'Orbitron', monospace;
    letter-spacing: 4px;
}

.sub-title {
    text-align: center;
    color: #22d3ee;
    font-size: 1.1rem;
    letter-spacing: 6px;
    margin-bottom: 0.25rem;
}

.tagline {
    text-align: center;
    color: #6b7280;
    font-size: 0.75rem;
    letter-spacing: 3px;
    margin-bottom: 2rem;
}

/* Input textarea */
textarea, .gr-input, .gr-textbox textarea {
    background: rgba(17, 17, 40, 0.9) !important;
    border: 1px solid rgba(139, 92, 246, 0.5) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.2) !important;
    padding: 1rem !important;
}

textarea:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 25px rgba(34, 211, 238, 0.3) !important;
    outline: none !important;
}

label span {
    color: #a78bfa !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Button */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.5) !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    box-shadow: 0 0 35px rgba(139, 92, 246, 0.8) !important;
    transform: scale(1.03) !important;
}

/* Output box */
.gr-textbox-output textarea, .output-class {
    background: rgba(17, 17, 40, 0.9) !important;
    border-radius: 16px !important;
    color: #22d3ee !important;
    font-family: 'Orbitron', monospace !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #4b5563;
    font-size: 0.65rem;
    letter-spacing: 3px;
    margin-top: 1.5rem;
}

.footer-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #a78bfa;
    border-radius: 50%;
    margin: 0 8px;
    vertical-align: middle;
}
"""

def analyze(message: str) -> tuple[str, str, str, str]:
    """Run prediction and return formatted outputs."""
    if not message.strip():
        return "⚠️ Please enter a message to analyze.", "", "", ""

    result = predict_spam(message)

    # Verdict
    if result['is_spam']:
        verdict = "🚨 SPAM DETECTED"
    else:
        verdict = "✅ NOT SPAM"

    confidence = f"{result['confidence']}%"
    risk       = f"RISK LEVEL: {result['risk_level'].upper()}"
    engine     = f"ENGINE: {result['method'].upper()}"

    return verdict, confidence, risk, engine


# ── Build the Gradio UI ────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="Titan Spam Detector") as demo:

    gr.HTML("""
        <div class='main-title'>⚡ TITAN</div>
        <div class='sub-title'>SPAM DETECTOR 🚫</div>
        <div class='tagline'>/// Say No to spam with our new Titan Spam Detector ///</div>
    """)

    with gr.Row():
        msg_input = gr.Textbox(
            label="MESSAGE INPUT",
            placeholder=">>> Enter message for analysis...",
            lines=6,
            max_lines=10,
        )

    analyze_btn = gr.Button("⚡ ANALYZE THREAT", variant="primary")

    with gr.Row():
        verdict_out    = gr.Textbox(label="VERDICT",     interactive=False)
        confidence_out = gr.Textbox(label="CONFIDENCE",  interactive=False)

    with gr.Row():
        risk_out   = gr.Textbox(label="RISK LEVEL", interactive=False)
        engine_out = gr.Textbox(label="ENGINE",     interactive=False)

    analyze_btn.click(
        fn=analyze,
        inputs=[msg_input],
        outputs=[verdict_out, confidence_out, risk_out, engine_out],
    )

    # Allow Enter key via examples
    gr.Examples(
        examples=[
            ["Congratulations! You've won a FREE iPhone. Click here to claim now: http://win.xyz"],
            ["Hey, are we still meeting tomorrow at 3pm?"],
            ["URGENT: Your bank account has been suspended. Call 08001234567 immediately."],
            ["Happy birthday! Hope you have a great day 🎂"],
        ],
        inputs=msg_input,
        label="📋 Try an example",
    )

    gr.HTML("<div class='footer-text'><span class='footer-dot'></span>POWERED BY TEAM PROMPT TITANS<span class='footer-dot'></span></div>")


if __name__ == "__main__":
    demo.launch()

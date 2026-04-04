"""HTML templates and reusable UI components for frontend routes."""

from __future__ import annotations

import json
from typing import Any, Dict


def base_layout(title: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{title}</title>
    <script src=\"https://cdn.tailwindcss.com\"></script>
    <script src=\"https://unpkg.com/htmx.org@1.9.10\"></script>
    <style>
        .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .card {{ transition: all .2s ease; }}
        .card:hover {{ transform: translateY(-2px); }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }}
        .btn-secondary {{
            background: #f3f4f6;
            color: #374151;
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }}
        .alert-error {{ background: #fee; border: 1px solid #fcc; color: #c33; padding: 12px; border-radius: 8px; }}
        .alert-success {{ background: #efe; border: 1px solid #cfc; color: #3c3; padding: 12px; border-radius: 8px; }}
        .alert-warning {{ background: #fff8e6; border: 1px solid #f4d27a; color: #8a6d1d; padding: 12px; border-radius: 8px; }}
        .spinner {{ width: 22px; height: 22px; border: 3px solid #ddd; border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body class=\"gradient-bg min-h-screen flex items-center justify-center py-12 px-4\">
    <div class=\"w-full max-w-2xl\">{content}</div>
    <script>
      function copyToClipboard(text, element) {{
        navigator.clipboard.writeText(text).then(() => {{
          const original = element.textContent;
          element.textContent = '✓ Copied!';
          setTimeout(() => element.textContent = original, 1200);
        }});
      }}
    </script>
</body>
</html>"""


def upload_page() -> str:
    return """<div class=\"card bg-white rounded-2xl shadow-2xl p-8\">
  <div class=\"text-center mb-8\">
    <h1 class=\"text-4xl font-bold mb-2\">🚀 PRISM</h1>
    <p class=\"text-gray-600 text-sm\">Deploy ML models with one click</p>
  </div>

  <form hx-post=\"/api/upload-and-run-ui\" hx-target=\"#response\" hx-indicator=\"#loading\" enctype=\"multipart/form-data\" class=\"space-y-6\">
    <div>
      <label class=\"block text-sm font-medium text-gray-700 mb-3\">Select Model File</label>
      <input type=\"file\" name=\"file\" id=\"modelFile\" accept=\".onnx,.pkl,.pickle,.joblib\" required onchange=\"document.getElementById('fileName').textContent=this.files[0]?.name||'No file selected'\" class=\"w-full text-sm\">
      <p id=\"fileName\" class=\"text-xs text-gray-500 mt-2\">No file selected</p>
    </div>

    <div class=\"flex items-center gap-3 bg-gray-50 p-4 rounded-lg\">
      <input type=\"checkbox\" id=\"enableTunnel\" name=\"enable_tunnel\" class=\"w-5 h-5\">
      <label for=\"enableTunnel\" class=\"text-sm font-medium text-gray-700\">Enable Public Tunnel</label>
      <span class=\"text-xs text-gray-500 ml-auto\">Share prediction link publicly</span>
    </div>

    <button type=\"submit\" class=\"btn-primary w-full\">Upload & Deploy</button>
  </form>

  <div id=\"loading\" class=\"htmx-indicator text-center py-8\">
    <div class=\"spinner mx-auto mb-3\"></div>
    <p class=\"text-gray-600 text-sm\">Deploying your model...</p>
  </div>

  <div id=\"response\" class=\"mt-8\"></div>
</div>"""


def upload_success_response(
  model_id: str,
  port: int,
  tunnel_url: str | None = None,
  tunnel_warning: str | None = None,
) -> str:
    ui_url = f"http://127.0.0.1:8000/predict?model_id={model_id}"
    api_url = f"http://127.0.0.1:8000/models/{model_id}/predict"
    public_block = ""
    if tunnel_url:
        public_block = f"""
    <div>
      <p class=\"text-sm font-medium text-gray-700 mb-2\">Public Tunnel URL 🌐</p>
      <div class=\"flex gap-2 items-center\">
        <code class=\"flex-1 bg-white p-3 rounded border border-green-200 bg-green-50 text-sm font-mono text-gray-900 overflow-auto\">{tunnel_url}</code>
        <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{tunnel_url}', this)\">Copy</button>
      </div>
      <p class=\"text-xs text-gray-600 mt-2\">Share this link with anyone to let them send predictions without exposing your local IP.</p>
    </div>
"""

    warning_block = ""
    if tunnel_warning:
      warning_block = (
        f'<div class="alert-warning mb-4">⚠ Tunnel unavailable: {tunnel_warning}</div>'
      )

    return f"""<div class=\"alert-success mb-4\">✓ Model deployed successfully!</div>
  {warning_block}
<div class=\"space-y-4 bg-gray-50 rounded-lg p-6\">
  <div>
    <p class=\"text-sm font-medium text-gray-700 mb-2\">Model ID</p>
    <code class=\"block bg-white p-3 rounded border border-gray-200 text-sm font-mono text-gray-900 break-all\">{model_id}</code>
  </div>
  <div>
    <p class=\"text-sm font-medium text-gray-700 mb-2\">Local Prediction URL (UI)</p>
    <div class=\"flex gap-2 items-center\">
      <code class=\"flex-1 bg-white p-3 rounded border border-gray-200 text-sm font-mono text-gray-900 overflow-auto\">{ui_url}</code>
      <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{ui_url}', this)\">Copy</button>
    </div>
  </div>
  <div>
    <p class=\"text-sm font-medium text-gray-700 mb-2\">Local Prediction API</p>
    <div class=\"flex gap-2 items-center\">
      <code class=\"flex-1 bg-white p-3 rounded border border-gray-200 text-sm font-mono text-gray-900 overflow-auto\">{api_url}</code>
      <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{api_url}', this)\">Copy</button>
    </div>
  </div>
  {public_block}
</div>
<div class=\"mt-6 flex gap-3\">
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload Another</a>
</div>"""


def predict_page() -> str:
    return """<div class=\"card bg-white rounded-2xl shadow-2xl p-8\">
  <div class=\"text-center mb-8\">
    <h1 class=\"text-3xl font-bold mb-2\">🔮 Make Predictions</h1>
    <p class=\"text-gray-600 text-sm\">Send input data to your deployed model</p>
  </div>

  <div class=\"bg-gray-50 p-4 rounded-lg mb-6\"><p class=\"text-sm text-gray-600\"><strong id=\"modelIdDisplay\">Loading...</strong></p></div>

  <form hx-post=\"/predict-result\" hx-target=\"#result\" hx-indicator=\"#predictLoading\" class=\"space-y-6\">
    <input type=\"hidden\" id=\"modelIdInput\" name=\"model_id\">
    <div>
      <label class=\"block text-sm font-medium text-gray-700 mb-3\">Input Data (JSON)</label>
      <textarea name=\"input_data\" id=\"inputData\" placeholder='Example: {\"age\": 25, \"salary\": 50000}' required class=\"w-full px-4 py-3 border border-gray-300 rounded-lg font-mono text-sm h-32 resize-none\"></textarea>
      <p class=\"text-xs text-gray-500 mt-2\">Enter your input as JSON. Check the model's requirements for correct format.</p>
    </div>
    <button type=\"submit\" class=\"btn-primary w-full\">Get Prediction</button>
  </form>

  <div id=\"predictLoading\" class=\"htmx-indicator text-center py-8\"><div class=\"spinner mx-auto mb-3\"></div><p class=\"text-gray-600 text-sm\">Processing your request...</p></div>
  <div id=\"result\" class=\"mt-8\"></div>
</div>
<script>
  const urlParams = new URLSearchParams(window.location.search);
  const modelId = urlParams.get('model_id');
  if (modelId) {
    document.getElementById('modelIdInput').value = modelId;
    document.getElementById('modelIdDisplay').textContent = 'Model: ' + modelId;
  } else {
    document.getElementById('modelIdDisplay').innerHTML = '<span class=\"text-red-600\">Model ID not provided. <a href=\"/\" class=\"underline\">Go back to upload.</a></span>';
  }
</script>"""


def prediction_result_component(prediction: Dict[str, Any], model_id: str) -> str:
    pretty = json.dumps(prediction, indent=2)
    return f"""<div class=\"alert-success mb-4\">✓ Prediction completed!</div>
<div class=\"bg-gray-900 p-6 rounded-lg\">
  <p class=\"text-gray-400 text-xs mb-2\">Response (JSON):</p>
  <pre class=\"text-green-400 font-mono text-sm overflow-auto max-h-48\">{pretty}</pre>
</div>
<div class=\"mt-6 flex gap-3\">
  <button hx-get=\"/predict?model_id={model_id}\" hx-target=\"body\" hx-replace=\"outerHTML swap:1s\" class=\"btn-secondary flex-1\">Make Another Prediction</button>
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload New Model</a>
</div>"""


def prediction_error_component(error: str, model_id: str) -> str:
    return f"""<div class=\"alert-error mb-4\">✗ Prediction failed: {error}</div>
<div class=\"mt-6 flex gap-3\">
  <button hx-get=\"/predict?model_id={model_id}\" hx-target=\"body\" hx-replace=\"outerHTML swap:1s\" class=\"btn-secondary flex-1\">Try Again</button>
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload New Model</a>
</div>"""

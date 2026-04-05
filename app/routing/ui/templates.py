"""HTML templates and reusable UI components for frontend routes."""

from __future__ import annotations

import json
from typing import Any, Dict

from app.utils.docker_utils import escape_html


def base_layout(title: str, content: str, show_sidebar: bool = False, active_nav: str = "") -> str:
    sidebar = ""
    if show_sidebar:
        dashboard_class = (
            "block px-4 py-3 rounded-md transition-colors bg-black text-white hover:opacity-90 font-medium text-sm"
            if active_nav == "dashboard"
            else "block px-4 py-3 rounded-md transition-colors hover:bg-gray-100 text-black font-medium text-sm border border-black"
        )
        upload_class = (
            "block px-4 py-3 rounded-md transition-colors bg-black text-white hover:opacity-90 font-medium text-sm"
            if active_nav == "upload"
            else "block px-4 py-3 rounded-md transition-colors hover:bg-gray-100 text-black font-medium text-sm border border-black"
        )
        logs_class = (
            "block px-4 py-3 rounded-md transition-colors bg-black text-white hover:opacity-90 font-medium text-sm"
            if active_nav == "logs"
            else "block px-4 py-3 rounded-md transition-colors hover:bg-gray-100 text-black font-medium text-sm border border-black"
        )

        sidebar = f"""
    <div class="fixed left-0 top-0 w-64 h-screen bg-white text-black shadow-lg flex flex-col border-r-2 border-black">
        <div class="p-6 border-b-2 border-black">
            <h1 class="text-2xl font-bold">🚀 PRISM</h1>
            <p class="text-xs text-gray-600 mt-1">Model Control Center</p>
        </div>
        <nav class="flex-1 p-4 space-y-3">
            <a href="/" class="{dashboard_class}">📊 Dashboard</a>
            <a href="/upload-model" class="{upload_class}">📤 Upload Model</a>
            <a href="/model-logs" class="{logs_class}">📋 Model Logs</a>
        </nav>
        <div class="p-4 border-t-2 border-black text-xs text-gray-700">
            <p>v1.0 Beta</p>
        </div>
    </div>
    <div class="ml-64">
    """

    closing_div = "</div>" if show_sidebar else ""

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{title}</title>
    <script src=\"https://cdn.tailwindcss.com\"></script>
    <script src=\"https://unpkg.com/htmx.org@1.9.10\"></script>
    <script src=\"https://unpkg.com/htmx.org/dist/ext/remove-me.js\"></script>
    <style>
        .gradient-bg {{ background: #000000; }}
        .card {{ transition: border-color .12s ease, box-shadow .12s ease; border-radius: 8px; }}
        .card:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .btn-primary {{
            background: #000000;
            color: #ffffff;
            padding: 12px 24px;
            border: 2px solid #000000;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color .12s ease, color .12s ease;
        }}
        .btn-primary:hover {{ background: #1a1a1a; border-color: #1a1a1a; }}
        .btn-secondary {{
            background: #ffffff;
            color: #000000;
            padding: 8px 16px;
            border: 1px solid #000000;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            text-decoration: none;
            display: inline-block;
            text-align: center;
            transition: all .2s ease;
        }}
        .btn-secondary:hover {{ background: #f3f4f6; }}
        .btn-danger {{ background: #ffffff; color: #000000; padding: 8px 16px; border: 1px solid #000000; border-radius: 4px; cursor: pointer; font-size: 13px; }}
        .btn-danger:hover {{ background: #f5f5f5; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid; }}
        .status-running {{ background: #ffffff; color: #000000; border-color: #000000; }}
        .status-stopped {{ background: #f5f5f5; color: #666666; border-color: #999999; }}
        .alert-error {{ background: #ffffff; border: 2px solid #000000; color: #000000; padding: 12px; border-radius: 6px; }}
        .alert-success {{ background: #ffffff; border: 2px solid #000000; color: #000000; padding: 12px; border-radius: 6px; }}
        .alert-warning {{ background: #ffffff; border: 2px solid #000000; color: #000000; padding: 12px; border-radius: 6px; }}
        .alert-info {{ background: #ffffff; border: 2px solid #000000; color: #000000; padding: 12px; border-radius: 6px; }}
        .spinner {{ width: 22px; height: 22px; border: 3px solid #cccccc; border-top-color: #000000; border-radius: 50%; animation: spin 1s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .model-card {{ background: #ffffff; border-radius: 8px; padding: 20px; border: 1px solid #111111; transition: border-color .12s ease, box-shadow .12s ease; }}
        .model-card:hover {{ border-color: #000000; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
        .status-indicator {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; border: 1px solid; }}
        .status-indicator.running {{ background: #000000; border-color: #000000; }}
        .status-indicator.stopped {{ background: #ffffff; border-color: #000000; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .5; }} }}
    </style>
</head>
<body {("class=\"bg-gray-50\"" if show_sidebar else "class=\"bg-white min-h-screen flex items-center justify-center py-12 px-4\"")}>
    {sidebar}
  <div id="app-content" {("class=\"p-8\"" if show_sidebar else "class=\"w-full max-w-2xl\"")}>
        {content}
    </div>
    {closing_div}
    <script>
      function copyToClipboard(text, element) {{
        navigator.clipboard.writeText(text).then(() => {{
          const original = element.textContent;
          element.textContent = '✓ Copied!';
          setTimeout(() => element.textContent = original, 1200);
        }});
      }}
      
      // HTMX confirmation handler
      htmx.on('htmx:confirm', function(e) {{
        if (!window.confirm(e.detail.question)) {{
          e.preventDefault();
        }}
      }});
    </script>
</body>
</html>"""


def upload_page() -> str:
    return """<div class=\"card bg-white rounded-lg shadow-md p-8 border border-black\">
  <div class=\"text-center mb-8 border-b border-black pb-6\">
    <h1 class=\"text-4xl font-bold mb-2\">🚀 PRISM</h1>
    <p class=\"text-gray-700 text-sm\">Deploy ML models with one click</p>
  </div>

  <form hx-post=\"/api/upload-and-run-ui\" hx-target=\"#response\" hx-indicator=\"#loading\" enctype=\"multipart/form-data\" class=\"space-y-6\">
    <div>
      <label class=\"block text-sm font-medium text-black mb-3\">Select Model File</label>
      <input type=\"file\" name=\"file\" id=\"modelFile\" accept=\".onnx,.pkl,.pickle,.joblib\" required onchange=\"document.getElementById('fileName').textContent=this.files[0]?.name||'No file selected'\" class=\"w-full text-sm border border-black p-3 rounded-md\">
      <p id=\"fileName\" class=\"text-xs text-gray-600 mt-2\">No file selected</p>
    </div>

    <div class=\"flex items-center gap-3 bg-white p-4 rounded-md border border-black\">
      <input type=\"checkbox\" id=\"enableTunnel\" name=\"enable_tunnel\" class=\"w-5 h-5 accent-black\">
      <label for=\"enableTunnel\" class=\"text-sm font-medium text-black\">Enable Public Tunnel</label>
      <span class=\"text-xs text-gray-600 ml-auto\">Share prediction link publicly</span>
    </div>

    <button type=\"submit\" class=\"btn-primary w-full\">Upload & Deploy</button>
  </form>

  <div id=\"loading\" class=\"htmx-indicator text-center py-8\">
    <div class=\"spinner mx-auto mb-3\"></div>
    <p class=\"text-gray-700 text-sm\">Deploying your model...</p>
  </div>

  <div id=\"response\" class=\"mt-8\"></div>
</div>"""


def upload_success_response(
  model_id: str,
  port: int,
  tunnel_url: str | None = None,
  tunnel_warning: str | None = None,
  qr_data_uri: str | None = None,
) -> str:
    ui_url = f"http://127.0.0.1:8000/predict?model_id={model_id}"
    api_url = f"http://127.0.0.1:8000/models/{model_id}/predict"
    public_block = ""
    qr_section = ""
    if tunnel_url:
        qr_for_tunnel = ""
        if qr_data_uri:
            qr_for_tunnel = f"""
      <div class=\"flex justify-center mt-3\">
        <img src=\"{qr_data_uri}\" alt=\"QR Code for public URL\" class=\"border border-black\" style=\"width:200px;height:200px;\">
      </div>
"""
        public_block = f"""
    <div>
      <p class=\"text-sm font-medium text-black mb-2\">Public Tunnel URL 🌐</p>
      <div class=\"flex gap-2 items-center\">
        <code class=\"flex-1 bg-white p-3 rounded border border-black text-sm font-mono text-black overflow-auto\">{tunnel_url}</code>
        <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{tunnel_url}', this)\">Copy</button>
      </div>
      <p class=\"text-xs text-gray-600 mt-2\">Share this link with anyone to let them send predictions without exposing your local IP.</p>
      {qr_for_tunnel}
    </div>
"""

    warning_block = ""
    if tunnel_warning:
      warning_block = (
        f'<div class=\"alert-warning mb-4\">⚠ Tunnel unavailable: {tunnel_warning}</div>'
      )

    return f"""<div class=\"alert-success mb-4\">✓ Model deployed successfully!</div>
  {warning_block}
<div class=\"space-y-4 bg-white rounded-lg p-6 border border-black\">
  <div>
    <p class=\"text-sm font-medium text-black mb-2\">Model ID</p>
    <code class=\"block bg-white p-3 rounded border border-black text-sm font-mono text-black break-all\">{model_id}</code>
  </div>
  <div>
    <p class=\"text-sm font-medium text-black mb-2\">Local Prediction URL (UI)</p>
    <div class=\"flex gap-2 items-center\">
      <code class=\"flex-1 bg-white p-3 rounded border border-black text-sm font-mono text-black overflow-auto\">{ui_url}</code>
      <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{ui_url}', this)\">Copy</button>
    </div>
  </div>
  <div>
    <p class=\"text-sm font-medium text-black mb-2\">Local Prediction API</p>
    <div class=\"flex gap-2 items-center\">
      <code class=\"flex-1 bg-white p-3 rounded border border-black text-sm font-mono text-black overflow-auto\">{api_url}</code>
      <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{api_url}', this)\">Copy</button>
    </div>
  </div>
  {public_block}
</div>
<div class=\"mt-6 flex gap-3\">
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload Another</a>
</div>"""


def _build_sample_from_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
      return {}

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
      result: dict[str, Any] = {}
      properties = schema.get("properties", {})
      required = schema.get("required", [])
      if isinstance(properties, dict):
        keys = list(properties.keys())
        for key in keys:
          value_schema = properties.get(key, {})
          if key in required or len(result) < 3:
            result[key] = _build_sample_from_schema(value_schema)
      return result

    if schema_type == "array":
      item_schema = schema.get("items", {})
      return [_build_sample_from_schema(item_schema)]

    if schema_type == "string":
      return "text"
    if schema_type == "integer":
      return 1
    if schema_type == "number":
      return 1.0
    if schema_type == "boolean":
      return True
    if schema_type == "null":
      return None

    return {}


def _contract_hints(expected_input_json: str | None) -> tuple[str, str | None]:
    if not expected_input_json:
      return "", None

    try:
      parsed = json.loads(expected_input_json)
    except json.JSONDecodeError:
      return "", None

    if not isinstance(parsed, dict):
      return "", json.dumps(parsed, indent=2, ensure_ascii=False)

    schema_like = any(
      key in parsed
      for key in ("type", "properties", "required", "items", "enum", "additionalProperties")
    )
    if not schema_like:
      return "", json.dumps(parsed, indent=2, ensure_ascii=False)

    hints: list[str] = []
    required = parsed.get("required", [])
    if isinstance(required, list) and required:
      hints.append("Required fields: " + ", ".join(str(item) for item in required))

    properties = parsed.get("properties", {})
    if isinstance(properties, dict) and properties:
      typed_fields: list[str] = []
      for field_name, field_schema in properties.items():
        field_type = "any"
        if isinstance(field_schema, dict) and isinstance(field_schema.get("type"), str):
          field_type = field_schema["type"]
        typed_fields.append(f"{field_name}: {field_type}")
      hints.append("Field types: " + ", ".join(typed_fields))

    additional = parsed.get("additionalProperties")
    if additional is False:
      hints.append("Additional fields are not allowed")

    sample_payload = _build_sample_from_schema(parsed)
    sample_json = json.dumps(sample_payload, indent=2, ensure_ascii=False)

    if not hints:
      return "", sample_json

    hints_html = "".join(
      f"<li class=\"text-xs text-gray-700\">{escape_html(hint)}</li>" for hint in hints
    )
    return f"<ul class=\"list-disc list-inside space-y-1 mt-2\">{hints_html}</ul>", sample_json


def predict_page(
    model_name: str | None = None,
    model_description: str | None = None,
    expected_input_json: str | None = None,
) -> str:
    hints_html, sample_json = _contract_hints(expected_input_json)
    model_meta_block = ""
    if model_name or model_description or expected_input_json:
        name_line = f"<p class=\"text-sm text-black mb-1\"><strong>Name:</strong> {escape_html(model_name or '')}</p>" if model_name else ""
        description_line = f"<p class=\"text-sm text-gray-700 mb-1\"><strong>Description:</strong> {escape_html(model_description or '')}</p>" if model_description else ""
        expected_line = ""
        if expected_input_json:
            expected_line = (
                "<p class=\"text-sm text-black mb-1\"><strong>Expected Input JSON:</strong></p>"
                f"<pre class=\"text-xs text-black bg-white border border-black rounded-md p-3 overflow-auto\">{escape_html(expected_input_json)}</pre>"
                + ("<p class=\"text-sm text-black mt-3 mb-1\"><strong>Input Contract Hints:</strong></p>" + hints_html if hints_html else "")
                + (f"<p class=\"text-sm text-black mt-3 mb-1\"><strong>Sample Payload:</strong></p><pre class=\"text-xs text-black bg-white border border-black rounded-md p-3 overflow-auto\">{escape_html(sample_json)}</pre>" if sample_json else "")
            )
        model_meta_block = (
            "<div class=\"bg-white p-4 rounded-md border border-black mb-6 space-y-1\">"
            f"{name_line}{description_line}{expected_line}"
            "</div>"
        )

    return """<div class=\"card bg-white rounded-lg shadow-md p-8 border border-black\">
  <div class=\"text-center mb-8 border-b border-black pb-6\">
    <h1 class=\"text-3xl font-bold mb-2\">🔮 Make Predictions</h1>
    <p class=\"text-gray-700 text-sm\">Send input data to your deployed model</p>
  </div>

  <div class=\"bg-white p-4 rounded-md border border-black mb-6\"><p class=\"text-sm text-black\"><strong id=\"modelIdDisplay\">Loading...</strong></p></div>
  """ + model_meta_block + """

  <form hx-post=\"/predict-result\" hx-target=\"#result\" hx-indicator=\"#predictLoading\" class=\"space-y-6\">
    <input type=\"hidden\" id=\"modelIdInput\" name=\"model_id\">
    <div>
      <label class=\"block text-sm font-medium text-black mb-3\">Input Data (JSON)</label>
      <textarea name=\"input_data\" id=\"inputData\" placeholder='Example: {\"age\": 25, \"salary\": 50000}' required class=\"w-full px-4 py-3 border border-black rounded-md font-mono text-sm h-32 resize-none\"></textarea>
      <p class=\"text-xs text-gray-600 mt-2\">Enter your input as JSON. Check the model's requirements for correct format.</p>
    </div>
    <button type=\"submit\" class=\"btn-primary w-full\">Get Prediction</button>
  </form>

  <div id=\"predictLoading\" class=\"htmx-indicator text-center py-8\"><div class=\"spinner mx-auto mb-3\"></div><p class=\"text-gray-700 text-sm\">Processing your request...</p></div>
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
<div class=\"bg-white p-6 rounded-lg border border-black\">
  <p class=\"text-black text-xs mb-2 font-medium\">Response (JSON):</p>
  <pre class=\"text-black font-mono text-sm overflow-auto max-h-48 bg-white p-3 border border-black rounded-md\">{pretty}</pre>
</div>
<div class=\"mt-6 flex gap-3\">
  <button hx-get=\"/predict?model_id={model_id}\" hx-target=\"#app-content\" hx-swap=\"innerHTML\" class=\"btn-secondary flex-1\">Make Another Prediction</button>
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload New Model</a>
</div>"""


def prediction_error_component(error: str, model_id: str) -> str:
    return f"""<div class=\"alert-error mb-4\">✗ Prediction failed: {error}</div>
<div class=\"mt-6 flex gap-3\">
  <button hx-get=\"/predict?model_id={model_id}\" hx-target=\"#app-content\" hx-swap=\"innerHTML\" class=\"btn-secondary flex-1\">Try Again</button>
  <a href=\"/\" class=\"btn-secondary flex-1 text-center\">Upload New Model</a>
</div>"""


def dashboard_page_with_cards(model_cards: list, has_models: bool) -> str:
    """Render dashboard with model card DTOs."""
    if not has_models:
        empty_state = """
    <div class=\"text-center py-12\">
        <div class=\"text-6xl mb-4\">📦</div>
        <h2 class=\"text-2xl font-bold text-black mb-2\">No Models Deployed</h2>
        <p class=\"text-gray-700 mb-6\">Get started by uploading your first ML model</p>
        <a href=\"/upload-model\" class=\"btn-primary inline-block\">Upload Model</a>
    </div>
"""
        return empty_state

    card_html = []
    for card in model_cards:
        tunnel_block = ""
        if card.tunnel_url:
            tunnel_block = f'''
            <div class="bg-white border border-black rounded-lg p-3 mt-4">
                <p class="text-xs font-medium text-black mb-2">🌐 Public Tunnel</p>
                <code class="text-xs bg-white p-2 rounded border border-black block overflow-auto mb-2 font-mono">{card.tunnel_url}</code>
                <button type="button" class="btn-secondary text-xs" onclick="copyToClipboard('{card.tunnel_url}', this)">Copy Link</button>
            </div>
'''

        card_html.append(f"""
    <div class="model-card">
        <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
                <div class="flex items-center gap-2 mb-2">
                    <span class="status-indicator {card.indicator_class}\"></span>
              <h3 class="font-bold text-lg text-black break-all">{card.model_name}</h3>
                </div>
            <p class="text-xs text-gray-600 mb-2">ID: {card.model_id}</p>
            {f'<p class="text-sm text-gray-700 mb-2">{escape_html(card.description)}</p>' if card.description else ''}
            {f'<details class="mb-2"><summary class="text-xs text-black cursor-pointer">Expected Input JSON</summary><pre class="text-xs text-black bg-white border border-black rounded-md p-2 mt-2 overflow-auto">{escape_html(card.expected_input_json)}</pre></details>' if card.expected_input_json else ''}
                <span class="status-badge {card.status_class}">{card.status_text}</span>
            </div>
            <button hx-post="/api/restart-model" hx-vals='{{"container_id": "{card.container_id}"}}' hx-target="closest .model-card" hx-swap="outerHTML" class="btn-secondary text-xs whitespace-nowrap">🔄 Restart</button>
        </div>

        <div class="bg-white p-3 rounded-md border border-black mb-4 text-sm space-y-2">
            <div><span class="text-gray-700">Container:</span> <code class="bg-white px-2 py-1 rounded text-xs font-mono border border-black">{card.container_id[:12]}</code></div>
            <div><span class="text-gray-700">Port:</span> <span class="font-mono text-black">{card.port}</span></div>
        </div>

        <div class="space-y-2 mb-4">
            <button type="button" class="w-full btn-secondary text-sm text-left" onclick="copyToClipboard('{card.predict_url}', this)">
                <span class="text-xs">UI Prediction URL</span><br><code class="text-xs font-mono break-all">{card.predict_url}</code>
            </button>
            <button type="button" class="w-full btn-secondary text-sm text-left" onclick="copyToClipboard('{card.api_url}', this)">
                <span class="text-xs">API Endpoint</span><br><code class="text-xs font-mono break-all">{card.api_url}</code>
            </button>
        </div>

        {tunnel_block}

        <div class="flex gap-2 mt-4 pt-4 border-t border-gray-200">
            <a href="/predict?model_id={card.model_id}" class="btn-secondary flex-1 text-center text-xs">Predict</a>
            <button hx-get="/api/model-logs?container_id={card.container_id}" hx-target="#modal-logs-container" hx-swap="innerHTML" class="btn-secondary flex-1 text-xs" data-modal="logs">View Logs</button>
            <button hx-delete="/api/delete-model" hx-vals='{{"model_id": "{card.model_id}", "container_id": "{card.container_id}"}}' hx-target="closest .model-card" hx-swap="outerHTML" hx-confirm="Delete this model and its container? This cannot be undone." class="btn-danger flex-1 text-xs">Delete</button>
        </div>
    </div>
""")

    return f"""
<div class="mb-8">
    <h1 class="text-3xl font-bold text-black mb-2">📊 Control Center</h1>
    <p class="text-gray-700">Manage and monitor your deployed ML models</p>
</div>

<div class="bg-white rounded-lg p-4 mb-8 border-2 border-black flex items-center justify-between">
    <p class="text-sm text-black"><strong>Tip:</strong> Click on a model card to expand details or use the buttons above to upload new models or check logs.</p>
    {('<button hx-delete="/api/kill-all-models" hx-target="#app-content" hx-swap="innerHTML" hx-confirm="⚠️ WARNING: This will delete ALL models and stop all containers! This action cannot be undone. Are you sure?" class="btn-danger text-xs whitespace-nowrap">🔥 Kill All</button>' if has_models else '')}
</div>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    {"".join(card_html)}
</div>

<div id="modal-logs-container" class="hidden"></div>
"""


def model_logs_modal(container_id: str) -> str:
    """Render modal with container logs."""
    from app.services.dashboard_service import ContainerLogsService
    
    logs_dto = ContainerLogsService.get_container_logs_dto(container_id, lines=50)
    
    if logs_dto.has_error:
        logs_content = f"Error retrieving logs: {logs_dto.error}"
    else:
        logs_content = logs_dto.logs
    
    safe_logs = logs_content.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
  <div class="bg-white rounded-lg max-w-2xl w-full max-h-96 flex flex-col border border-black">
    <div class="flex items-center justify-between p-6 border-b border-black">
      <h2 class="text-xl font-bold text-black">Container Logs</h2>
      <button type="button" class="text-gray-700 hover:text-black" onclick="this.closest('.fixed').remove()">✕</button>
        </div>
    <div class="flex-1 overflow-auto bg-white p-4">
      <pre class="text-black font-mono text-xs whitespace-pre-wrap break-words border border-black rounded-md p-3">{safe_logs}</pre>
        </div>
    <div class="p-4 border-t border-black flex gap-2">
            <button type="button" class="btn-secondary flex-1" onclick="navigator.clipboard.writeText(`{logs_content.replace(chr(96), chr(92)+chr(96))}`).then(() => alert('Logs copied!'))">Copy Logs</button>
            <button type="button" class="btn-secondary flex-1" onclick="this.closest('.fixed').remove()">Close</button>
        </div>
    </div>
</div>
"""


def upload_model_page() -> str:
    """Render upload model page with sidebar."""
    return """
<div class="card bg-white rounded-lg shadow-md p-8 max-w-2xl mx-auto border border-black">
  <h1 class="text-3xl font-bold mb-2">📤 Upload New Model</h1>
  <p class="text-gray-700 text-sm mb-8">Deploy a new ML model to your control center</p>

  <form hx-post="/api/upload-and-run-ui" hx-target="#response" hx-indicator="#loading" enctype="multipart/form-data" class="space-y-6">
    <div>
      <label class="block text-sm font-medium text-black mb-3">Model Name</label>
      <input type="text" name="model_name" maxlength="120" placeholder="e.g. Customer Churn Predictor" class="w-full text-sm border border-black p-3 rounded-md">
    </div>

    <div>
      <label class="block text-sm font-medium text-black mb-3">Description</label>
      <textarea name="model_description" maxlength="500" placeholder="Short description of what this model predicts" class="w-full text-sm border border-black p-3 rounded-md h-24 resize-none"></textarea>
    </div>

    <div>
      <label class="block text-sm font-medium text-black mb-3">Expected Input JSON Format</label>
      <textarea name="expected_input_json" placeholder='{"input": [1.0, 2.0, 3.0]}' class="w-full text-sm border border-black p-3 rounded-md h-28 resize-none font-mono"></textarea>
      <p class="text-xs text-gray-600 mt-2">Optional but recommended. Must be valid JSON if provided.</p>
    </div>

    <div>
      <label class="block text-sm font-medium text-black mb-3">Select Model File</label>
      <input type="file" name="file" id="modelFile" accept=".onnx,.pkl,.pickle,.joblib" required onchange="document.getElementById('fileName').textContent=this.files[0]?.name||'No file selected'" class="w-full text-sm border border-black p-3 rounded-md">
      <p id="fileName" class="text-xs text-gray-600 mt-2">No file selected</p>
      <p class="text-xs text-gray-600 mt-2">Supported: ONNX (.onnx), Scikit-learn (.pkl, .pickle), joblib (.joblib)</p>
    </div>

    <div class="flex items-center gap-3 bg-white p-4 rounded-md border border-black">
      <input type="checkbox" id="enableTunnel" name="enable_tunnel" class="w-5 h-5 accent-black">
      <label for="enableTunnel" class="text-sm font-medium text-black">Enable Public Tunnel</label>
      <span class="text-xs text-gray-600 ml-auto">Share prediction link publicly</span>
    </div>

    <button type="submit" class="btn-primary w-full">Upload & Deploy</button>
  </form>

  <div id="loading" class="htmx-indicator text-center py-8">
    <div class="spinner mx-auto mb-3"></div>
    <p class="text-gray-700 text-sm">Deploying your model...</p>
  </div>

  <div id="response" class="mt-8"></div>

  <div class="mt-8 pt-8 border-t border-black">
    <p class="text-sm text-black mb-4"><strong>Need help?</strong></p>
    <ul class="text-sm text-gray-700 space-y-2 list-disc list-inside">
      <li>Your model must have a /predict endpoint</li>
      <li>Accept JSON input and return JSON output</li>
      <li>Maximum file size: 500MB</li>
    </ul>
  </div>
</div>
"""

"""HTML templates and reusable UI components for frontend routes."""

from __future__ import annotations

import json
from typing import Any, Dict

from app.utils.docker_utils import escape_html


def base_layout(
    title: str, content: str, show_sidebar: bool = False, active_nav: str = ""
) -> str:
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
        const question = e.detail && e.detail.question;
        if (!question) {{
          return;
        }}
        if (!window.confirm(question)) {{
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
    api_url = f"http://127.0.0.1:{port}/predict"
    public_block = ""
    if tunnel_url:
        tunnel_endpoint = f"{tunnel_url.rstrip('/')}/predict?model_id={model_id}"
        qr_for_tunnel = ""
        if qr_data_uri:
            qr_for_tunnel = f"""
      <div class=\"flex justify-center mt-3\">
        <img src=\"{qr_data_uri}\" alt=\"QR Code for public URL\" class=\"border border-black\" style=\"width:200px;height:200px;\">
      </div>
"""
        public_block = f"""
    <div>
      <p class=\"text-sm font-medium text-black mb-2\">Public Prediction URL 🌐</p>
      <div class=\"flex gap-2 items-center\">
        <code class=\"flex-1 bg-white p-3 rounded border border-black text-sm font-mono text-black overflow-auto\">{tunnel_endpoint}</code>
        <button type=\"button\" class=\"btn-secondary whitespace-nowrap\" onclick=\"copyToClipboard('{tunnel_endpoint}', this)\">Copy</button>
      </div>
      <p class=\"text-xs text-gray-600 mt-2\">Share this URL for the same prediction page flow as local UI.</p>
      {qr_for_tunnel}
    </div>
"""

    warning_block = ""
    if tunnel_warning:
        warning_block = f'<div class="alert-warning mb-4">⚠ Tunnel unavailable: {tunnel_warning}</div>'

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
        for key in (
            "type",
            "properties",
            "required",
            "items",
            "enum",
            "additionalProperties",
        )
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
            if isinstance(field_schema, dict) and isinstance(
                field_schema.get("type"), str
            ):
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
        f'<li class="text-xs text-gray-700">{escape_html(hint)}</li>' for hint in hints
    )
    return (
        f'<ul class="list-disc list-inside space-y-1 mt-2">{hints_html}</ul>',
        sample_json,
    )


def predict_page(
    model_name: str | None = None,
    model_description: str | None = None,
    expected_input_json: str | None = None,
    version: str | None = None,
) -> str:
    hints_html, sample_json = _contract_hints(expected_input_json)
    model_meta_block = ""
    if model_name or model_description or expected_input_json:
        name_line = (
            f"<p class=\"text-sm text-black mb-1\"><strong>Name:</strong> {escape_html(model_name or '')}</p>"
            if model_name
            else ""
        )
        description_line = (
            f"<p class=\"text-sm text-gray-700 mb-1\"><strong>Description:</strong> {escape_html(model_description or '')}</p>"
            if model_description
            else ""
        )
        expected_line = ""
        if expected_input_json:
            expected_line = '<p class="text-sm text-black mb-1"><strong>Expected Input JSON:</strong></p>' f'<pre class="text-xs text-black bg-white border border-black rounded-md p-3 overflow-auto">{escape_html(expected_input_json)}</pre>' + (
                '<p class="text-sm text-black mt-3 mb-1"><strong>Input Contract Hints:</strong></p>'
                + hints_html
                if hints_html
                else ""
            ) + (
                f'<p class="text-sm text-black mt-3 mb-1"><strong>Sample Payload:</strong></p><pre class="text-xs text-black bg-white border border-black rounded-md p-3 overflow-auto">{escape_html(sample_json)}</pre>'
                if sample_json
                else ""
            )
        model_meta_block = (
            '<div class="bg-white p-4 rounded-md border border-black mb-6 space-y-1">'
            f"{name_line}{description_line}{expected_line}"
            "</div>"
        )

    return (
        """<div class=\"card bg-white rounded-lg shadow-md p-8 border border-black\">
  <div class=\"text-center mb-8 border-b border-black pb-6\">
    <h1 class=\"text-3xl font-bold mb-2\">🔮 Make Predictions</h1>
    <p class=\"text-gray-700 text-sm\">Send input data to your deployed model</p>
  </div>

  <div class=\"bg-white p-4 rounded-md border border-black mb-6\"><p class=\"text-sm text-black\"><strong id=\"modelIdDisplay\">Loading...</strong></p></div>
  """
        + model_meta_block
        + """

  <form hx-post=\"/predict-result\" hx-target=\"#result\" hx-indicator=\"#predictLoading\" class=\"space-y-6\">
    <input type=\"hidden\" id=\"modelIdInput\" name=\"model_id\">
    <input type=\"hidden\" id=\"versionInput\" name=\"version\">
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
  const version = urlParams.get('version');
  if (modelId) {
    document.getElementById('modelIdInput').value = modelId;
    document.getElementById('modelIdDisplay').textContent = 'Model: ' + modelId;
  } else {
    document.getElementById('modelIdDisplay').innerHTML = '<span class=\"text-red-600\">Model ID not provided.</span>';
  }
  if (version) {
    document.getElementById('versionInput').value = version;
    document.getElementById('modelIdDisplay').textContent += ' (version: ' + version + ')';
  }
</script>"""
    )


def prediction_result_component(
    prediction: Dict[str, Any],
    model_id: str,
    version: str | None = None,
) -> str:
    pretty = json.dumps(prediction, indent=2)
    retry_url = f"/predict?model_id={model_id}"
    if version:
        retry_url += f"&version={version}"
    return f"""<div class=\"alert-success mb-4\">✓ Prediction completed!</div>
<div class=\"bg-white p-6 rounded-lg border border-black\">
  <p class=\"text-black text-xs mb-2 font-medium\">Response (JSON):</p>
  <pre class=\"text-black font-mono text-sm overflow-auto max-h-48 bg-white p-3 border border-black rounded-md\">{pretty}</pre>
</div>
<div class=\"mt-6\">
  <button hx-get=\"{retry_url}\" hx-target=\"#app-content\" hx-swap=\"innerHTML\" class=\"btn-secondary w-full\">Make Another Prediction</button>
</div>"""


def prediction_error_component(
    error: str,
    model_id: str,
    version: str | None = None,
) -> str:
    retry_url = f"/predict?model_id={model_id}"
    if version:
        retry_url += f"&version={version}"
    return f"""<div class=\"alert-error mb-4\">✗ Prediction failed: {error}</div>
<div class=\"mt-6\">
  <button hx-get=\"{retry_url}\" hx-target=\"#app-content\" hx-swap=\"innerHTML\" class=\"btn-secondary w-full\">Try Again</button>
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
            tunnel_block = f"""
            <div class="bg-white border border-black rounded-lg p-3 mt-4">
                <p class="text-xs font-medium text-black mb-2">🌐 Public Link (Tunnel)</p>
                <code class="text-xs bg-white p-2 rounded border border-black block overflow-auto mb-3 font-mono break-all">{card.tunnel_prediction_url}</code>
                <button type="button" class="btn-secondary text-xs w-full" onclick="copyToClipboard('{card.tunnel_prediction_url}', this)">Copy Link</button>
                <p class="text-xs text-gray-600 mt-2">Clients can ONLY access this model via this link.</p>
            </div>
"""

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
          <a href="/model/{card.model_id}/control" class="btn-secondary flex-1 text-xs">Control Center</a>
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

<div id="modal-logs-container"></div>
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

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium text-black mb-3">Version (optional)</label>
        <input type="text" name="version" maxlength="40" placeholder="e.g. v1.0" class="w-full text-sm border border-black p-3 rounded-md">
      </div>

      <div>
        <label class="block text-sm font-medium text-black mb-3">Replicas</label>
        <input type="number" name="replicas" min="1" max="10" value="1" class="w-full text-sm border border-black p-3 rounded-md">
        <p class="text-xs text-gray-600 mt-2">Deploy multiple instances for simple load balancing.</p>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium text-black mb-3">Load Balancing Strategy</label>
        <select name="load_balancing_strategy" class="w-full text-sm border border-black p-3 rounded-md">
          <option value="round-robin">Round-robin (default)</option>
          <option value="least-connections">Least-connections</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-black mb-3">Enable Request Caching</label>
        <div class="flex items-center gap-3">
          <input type="checkbox" id="enableCaching" name="enable_caching" class="w-5 h-5 accent-black">
          <label for="enableCaching" class="text-sm font-medium text-black">Enable</label>
        </div>
        <input type="number" name="cache_ttl" min="0" placeholder="TTL (seconds)" class="w-full text-sm border border-black p-3 rounded-md mt-2">
        <p class="text-xs text-gray-600 mt-2">Cache identical prediction inputs for this many seconds (0 = no TTL).</p>
      </div>
    </div>

    <div>
      <label class="block text-sm font-medium text-black mb-3">Enable Batching</label>
      <div class="flex items-center gap-3">
        <input type="checkbox" id="enableBatching" name="enable_batching" class="w-5 h-5 accent-black">
        <label for="enableBatching" class="text-sm font-medium text-black">Enable</label>
      </div>
      <div class="grid grid-cols-2 gap-3 mt-3">
        <input type="number" name="batch_size" min="1" placeholder="Batch size" class="text-sm border border-black p-3 rounded-md">
        <input type="number" name="batch_timeout" min="0" placeholder="Timeout (ms)" class="text-sm border border-black p-3 rounded-md">
      </div>
      <p class="text-xs text-gray-600 mt-2">Buffer up to N requests or wait M ms to form a batch.</p>
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

def model_control_center_page(model_id: str, model_entry: dict | None = None, version: str | None = None) -> str:
    model_name = model_entry.get('name') if isinstance(model_entry, dict) else model_id
    description = model_entry.get('description') if isinstance(model_entry, dict) else ''
    config = model_entry.get('config') if isinstance(model_entry, dict) else {}
    metrics_config = model_entry.get('metrics_config') if isinstance(model_entry, dict) else None
    if metrics_config is None:
        metrics_config = {}
    version_label = version or (model_entry.get('version') if isinstance(model_entry, dict) else None) or 'v1'
    
    # Get tunable parameters with defaults
    window_size = metrics_config.get('window_size', 60) if isinstance(metrics_config, dict) else 60
    update_interval_ms = metrics_config.get('update_interval_ms', 1000) if isinstance(metrics_config, dict) else 1000
    latency_threshold = metrics_config.get('latency_warning_threshold_ms', 1000) if isinstance(metrics_config, dict) else 1000
    error_threshold = metrics_config.get('error_rate_warning_threshold_pct', 5.0) if isinstance(metrics_config, dict) else 5.0
    chart_colors = (metrics_config.get('chart_colors', {
        'requests': '#000000',
        'latency': '#ff9900',
        'throughput': '#0066ff',
        'error_rate': '#ff0000',
        'cpu_usage': '#00cc00',
        'memory_usage': '#ff6600',
    }) if isinstance(metrics_config, dict) else {
        'requests': '#000000',
        'latency': '#ff9900',
        'throughput': '#0066ff',
        'error_rate': '#ff0000',
        'cpu_usage': '#00cc00',
        'memory_usage': '#ff6600',
    })
    
    return f"""
<div class="max-w-6xl mx-auto">
  <div class="flex items-center justify-between mb-6">
    <div>
      <h1 class="text-3xl font-bold text-black">Model Control Center</h1>
      <p class="text-gray-700">Overview and tunable metrics for <strong>{escape_html(model_name or '')}</strong> ({escape_html(model_id or '')})</p>
    </div>
    <div>
      <a href="/" class="btn-secondary">Back to Dashboard</a>
    </div>
  </div>

  <!-- Model Info Cards -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
    <div class="model-card">
      <p class="text-xs text-gray-600">Model ID</p>
      <p class="font-mono text-sm text-black break-all">{escape_html(model_id)}</p>
    </div>
    <div class="model-card">
      <p class="text-xs text-gray-600">Version</p>
      <p class="font-mono text-sm text-black">{escape_html(version_label)}</p>
    </div>
    <div class="model-card">
      <p class="text-xs text-gray-600">Description</p>
      <p class="text-sm text-gray-700">{escape_html(description or '')}</p>
    </div>
  </div>

  <!-- Metrics Configuration Controls -->
  <div class="card p-6 mb-6">
    <h2 class="text-lg font-bold text-black mb-4">⚙️ Metrics Configuration</h2>
    <form id="metricsConfigForm" hx-post="/api/metrics-config" hx-target="#configResponse">
      <input type="hidden" name="model_id" value="{escape_html(model_id)}">
      
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Display Window Size
            <span class="text-xs text-gray-500">(data points)</span>
          </label>
          <input 
            type="number" 
            name="window_size" 
            value="{window_size}"
            min="1" 
            max="600" 
            class="w-full px-3 py-2 border border-black rounded text-black"
          >
          <p class="text-xs text-gray-600 mt-1">Number of data points to display (1-600)</p>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Update Interval
            <span class="text-xs text-gray-500">(milliseconds)</span>
          </label>
          <input 
            type="number" 
            name="update_interval_ms" 
            value="{update_interval_ms}"
            min="100" 
            max="10000" 
            step="100"
            class="w-full px-3 py-2 border border-black rounded text-black"
          >
          <p class="text-xs text-gray-600 mt-1">How often to refresh metrics (100-10000 ms)</p>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Latency Warning Threshold
            <span class="text-xs text-gray-500">(milliseconds)</span>
          </label>
          <input 
            type="number" 
            name="latency_warning_threshold_ms" 
            value="{latency_threshold}"
            min="1" 
            class="w-full px-3 py-2 border border-black rounded text-black"
          >
          <p class="text-xs text-gray-600 mt-1">Warn when latency exceeds this threshold</p>
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Error Rate Warning Threshold
            <span class="text-xs text-gray-500">(%)</span>
          </label>
          <input 
            type="number" 
            name="error_rate_warning_threshold_pct" 
            value="{error_threshold}"
            min="0" 
            max="100" 
            step="0.1"
            class="w-full px-3 py-2 border border-black rounded text-black"
          >
          <p class="text-xs text-gray-600 mt-1">Warn when error rate exceeds this percentage</p>
        </div>
      </div>
      
      <div class="mt-6 flex gap-2">
        <button type="submit" class="btn-primary">Save Configuration</button>
        <button type="reset" class="btn-secondary">Reset</button>
      </div>
    </form>
    <div id="configResponse" class="mt-4"></div>
  </div>

  <!-- Performance Charts -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
    <div class="card p-4">
      <h3 class="text-sm font-semibold mb-2">📊 Requests</h3>
      <canvas id="requestsChart" height="160"></canvas>
    </div>
    <div class="card p-4">
      <h3 class="text-sm font-semibold mb-2">⏱️ Latency (ms)</h3>
      <canvas id="latencyChart" height="160"></canvas>
      <p class="text-xs text-gray-600 mt-2">⚠️ Warning threshold: {latency_threshold}ms</p>
    </div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
    <div class="card p-4">
      <h3 class="text-sm font-semibold mb-2">🚀 Throughput (req/s)</h3>
      <canvas id="throughputChart" height="160"></canvas>
    </div>
    <div class="card p-4">
      <h3 class="text-sm font-semibold mb-2">❌ Error Rate (%)</h3>
      <canvas id="errorChart" height="160"></canvas>
      <p class="text-xs text-gray-600 mt-2">⚠️ Warning threshold: {error_threshold}%</p>
    </div>
  </div>

  <!-- Deployment Configuration -->
  <div class="card p-4 mb-6">
    <h3 class="text-sm font-semibold mb-2">⚙️ Deployment Configuration</h3>
    <pre class="text-xs font-mono bg-white p-3 border border-black rounded text-gray-900">{escape_html(json.dumps(config or dict(), indent=2))}</pre>
  </div>

  <!-- Metrics Color Configuration -->
  <div class="card p-4 mb-6">
    <h3 class="text-sm font-semibold mb-2">🎨 Chart Colors</h3>
    <div class="grid grid-cols-2 lg:grid-cols-3 gap-4">
      <div class="text-sm">
        <p class="text-gray-600">Requests</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('requests', '#000000')};"></div>
        <code class="text-xs">{chart_colors.get('requests', '#000000')}</code>
      </div>
      <div class="text-sm">
        <p class="text-gray-600">Latency</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('latency', '#ff9900')};"></div>
        <code class="text-xs">{chart_colors.get('latency', '#ff9900')}</code>
      </div>
      <div class="text-sm">
        <p class="text-gray-600">Throughput</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('throughput', '#0066ff')};"></div>
        <code class="text-xs">{chart_colors.get('throughput', '#0066ff')}</code>
      </div>
      <div class="text-sm">
        <p class="text-gray-600">Error Rate</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('error_rate', '#ff0000')};"></div>
        <code class="text-xs">{chart_colors.get('error_rate', '#ff0000')}</code>
      </div>
      <div class="text-sm">
        <p class="text-gray-600">CPU Usage</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('cpu_usage', '#00cc00')};"></div>
        <code class="text-xs">{chart_colors.get('cpu_usage', '#00cc00')}</code>
      </div>
      <div class="text-sm">
        <p class="text-gray-600">Memory Usage</p>
        <div class="w-8 h-8 border-2 border-black rounded" style="background-color: {chart_colors.get('memory_usage', '#ff6600')};"></div>
        <code class="text-xs">{chart_colors.get('memory_usage', '#ff6600')}</code>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const MODEL_ID = '{escape_html(model_id)}';
    const CONFIG = {{
      windowSize: {window_size},
      updateInterval: {update_interval_ms},
      latencyThreshold: {latency_threshold},
      errorThreshold: {error_threshold},
      colors: {{
        requests: '{chart_colors.get('requests', '#000000')}',
        latency: '{chart_colors.get('latency', '#ff9900')}',
        throughput: '{chart_colors.get('throughput', '#0066ff')}',
        errorRate: '{chart_colors.get('error_rate', '#ff0000')}',
      }}
    }};
    
    let charts = {{}};
    
    async function fetchMetrics() {{
      try {{
        const res = await fetch(`/api/model-metrics?model_id=${{MODEL_ID}}`);
        return res.json();
      }} catch(e) {{
        console.error('Error fetching metrics:', e);
        return null;
      }}
    }}
    
    function renderChart(canvasId, labels, values, color, threshold = null) {{
      const ctx = document.getElementById(canvasId);
      if (!ctx) return;
      
      const options = {{
        responsive: true,
        maintainAspectRatio: true,
        scales: {{
          x: {{ display: false }},
          y: {{ beginAtZero: true }},
        }},
        plugins: {{
          legend: {{ display: false }},
        }}
      }};
      
      // Add threshold line if provided
      let plugins = undefined;
      if (threshold !== null) {{
        plugins = {{
          annotation: {{
            annotations: {{
              threshold: {{
                type: 'line',
                yMin: threshold,
                yMax: threshold,
                borderColor: '#ff0000',
                borderDash: [5, 5],
              }}
            }}
          }}
        }};
      }}
      
      if (charts[canvasId]) {{
        charts[canvasId].destroy();
      }}
      
      charts[canvasId] = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels,
          datasets: [{{
            data: values,
            borderColor: color,
            backgroundColor: color + '33',
            fill: true,
            tension: 0.2,
            borderWidth: 2,
          }}]
        }},
        options,
      }});
    }}
    
    async function updateCharts() {{
      const metrics = await fetchMetrics();
      if (!metrics) return;
      
      const labels = metrics.labels || [];
      const requests = metrics.requests || [];
      const latency = metrics.latency || [];
      const throughput = metrics.throughput || [];
      const errorRate = metrics.error_rate || [];
      
      renderChart('requestsChart', labels, requests, CONFIG.colors.requests);
      renderChart('latencyChart', labels, latency, CONFIG.colors.latency, CONFIG.latencyThreshold);
      renderChart('throughputChart', labels, throughput, CONFIG.colors.throughput);
      renderChart('errorChart', labels, errorRate, CONFIG.colors.errorRate, CONFIG.errorThreshold);
    }}
    
    // Initial load
    updateCharts();
    
    // Refresh on interval
    setInterval(updateCharts, CONFIG.updateInterval);
    
    // Allow manual refresh
    document.addEventListener('keydown', (e) => {{
      if (e.ctrlKey && e.key === 'r') {{
        e.preventDefault();
        updateCharts();
      }}
    }});
  </script>
</div>
"""

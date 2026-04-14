# Quick Start: Using the PRISM Dashboard

## 🚀 Getting Started

### 1. Start the Server
```bash
source /path/to/venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

The dashboard will be available at: **http://127.0.0.1:8000**

### 2. Navigate to Dashboard
Open your browser and go to `http://127.0.0.1:8000`

You'll see:
- 📊 Main dashboard with sidebar navigation
- Empty state (if no models deployed yet)
- Upload button in sidebar

## 📤 Deploy Your First Model

### Method 1: Using the UI
1. Click **"📤 Upload Model"** in the sidebar
2. Select your model file (.onnx, .pkl, .pickle, or .joblib)
3. (Optional) Check **"Enable Public Tunnel"** for public access
4. Click **"Upload & Deploy"**
5. See deployment details with copy-able URLs
6. Click **"Return to Dashboard"**

### Method 2: Using cURL
```bash
curl -X POST http://127.0.0.1:8000/api/upload-and-run-ui \
  -F "file=@my_model.pkl" \
  -F "enable_tunnel=false"
```

## 🔮 Make Predictions

### From Dashboard
1. Dashboard shows your deployed model
2. Click **"Predict"** button on the model card
3. Enter JSON input data:
   ```json
   {"age": 25, "salary": 50000}
   ```
4. Click **"Get Prediction"**
5. View results in pretty JSON format

### Using the API Directly
```bash
curl -X POST http://127.0.0.1:8000/models/{model_id}/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 25, "salary": 50000}'
```

## 📋 Monitor Model Logs

1. Click **"📋 Model Logs"** in sidebar
2. See list of all deployed models
3. Click any model to open logs modal
4. View last 50 lines of container output
5. Click **"Copy Logs"** to copy for analysis

## 🔄 Manage Containers

### Check Container Status
Dashboard shows real-time status with visual indicator:
- ✓ **Running** - Green pulsing indicator
- ✗ **Stopped** - Red static indicator

### Restart a Failed Container
1. Model shows "✗ Stopped" status
2. Click **"🔄 Restart"** button
3. Container restarts via Docker
4. Status updates to "✓ Running"

### Get Container Details
Each model card shows:
- **Container ID** (first 12 chars)
- **Host Port** (where container is mapped)
- **Tunnel URL** (if enabled)

## 🌐 Public Access with Tunnel

### Enable Tunnel During Upload
1. Upload model page -> Check "Enable Public Tunnel"
2. Requires `NGROK_AUTHTOKEN` env var set
3. Success response shows public URL

### Set ngrok Token
```bash
# One time setup
export NGROK_AUTHTOKEN="your_token_from_ngrok"

# Start server
python -m uvicorn app.main:app --port 8000
```

### Share Public URL
Success page shows:
```
Public Tunnel URL 🌐
https://abc-123-xyz.ngrok.io/predict?model_id=...
```

Copy this URL and share with anyone to make predictions!

## 📊 Dashboard Features Overview

### Model Card
```
┌─────────────────────────────────┐
│ ● Model: abc123 (green/red dot) │
│ Status: Running ✓               │
│ Container: prism_model_abc123   │
│ Port: 51234                     │
│                                 │
│ [Copy UI URL] [Copy API]        │
│ [Copy Public URL] (if enabled)  │
│                                 │
│ [Predict] [Logs] [Restart]      │
└─────────────────────────────────┘
```

### Sidebar Navigation
```
🚀 PRISM
Model Control Center
──────────────────
📊 Dashboard  (current page highlighted)
📤 Upload Model
📋 Model Logs
──────────────────
v1.0 Beta
```

## 🛠️ Advanced Usage

### Check Model API Endpoint
Every model has an API endpoint:
```
http://127.0.0.1:8000/models/{model_id}/predict
```

Send POST requests with JSON:
```bash
curl -X POST http://127.0.0.1:8000/models/abc123/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
```

### Debug Container Issues
1. Click **"View Logs"** on model card
2. Look for error messages
3. Common issues:
   - `UnpicklingError` - corrupted model file
   - `ModuleNotFoundError` - missing dependencies
   - `RuntimeError` - model loading error

### Restart Strategy
If model stops:
1. Check logs for error details
2. Fix model if needed
3. Restart container via dashboard
4. If still fails, upload new model

## 📚 Documentation

For more detailed information:
- **DASHBOARD.md** - Complete feature documentation
- **DASHBOARD_UI_GUIDE.md** - UI/UX design guide
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## ✅ Testing

Run tests to verify setup:
```bash
# Frontend tests only
python -m pytest tests/test_frontend.py -v

# All tests
python -m pytest tests/ -v

# Quick test
python -m pytest tests/ -q
```

Should see:
```
56 passed in 48.78s
```

## 🐛 Troubleshooting

### Dashboard won't load
- Check if server is running on port 8000
- Look for errors in server logs
- Try `http://127.0.0.1:8000` instead of localhost

### Upload fails
- Check file format (.onnx, .pkl, .pickle, .joblib)
- Ensure model has `/predict` endpoint
- Check available disk space
- Review error message in response

### Predictions return error
- Check model is running (green indicator)
- Verify input JSON matches model expectations
- Check logs for specific error
- Try restarting container

### Tunnel not working (ERR_NGROK_3200 or offline)
- **Ensure NGROK_AUTHTOKEN is set** - Required for any tunnel
  - `export NGROK_AUTHTOKEN="your_token_from_ngrok"`
  - Get token at https://ngrok.com (free account)
- **Wait for tunnel to stabilize** - Tunnels take 3-5 seconds to become fully active
  - The tunnel URL is created immediately but may not accept traffic initially
  - Logs show "Waiting for tunnel to stabilize" - this is expected
- **Check local endpoint is working** - Tunnel won't work if localhost:8000 is down
  - Verify app is running: `curl http://127.0.0.1:8000/`
  - If local endpoint fails, tunnel will also fail
- **Retry the request** - ngrok free tier may have intermittent connectivity
  - If tunnel URL returns offline error, wait 10-30 seconds and try again
  - Subsequent requests often succeed as ngrok stabilizes
- **Check ngrok region (optional)** - High-load regions may be congested
  - Set `NGROK_REGION=in` or `NGROK_REGION=eu` for different regions
  - Default region is `us`
- **Monitor tunnel status** - Check application logs for tunnel creation messages
  - Look for: `Successfully created tunnel: http://127.0.0.1:8000 -> https://...`
  - If test request fails but URL is created, tunnel may still work

### Logs won't load
- Ensure Docker is running
- Check `docker ps` shows model container
- Model may be too new (logs taking time)
- Try again in a few seconds

## 🎯 Common Workflows

### Deploy → Predict → Monitor
```
1. Click "Upload Model"
2. Select file, click "Upload & Deploy"
3. Copy URLs from response
4. Click "Return to Dashboard"
5. Click "Predict" on model card
6. Enter input data
7. Get prediction result
```

### Monitor Multiple Models
```
1. Dashboard shows all models
2. Scan status indicators
3. Click "Logs" for any troublesome model
4. Click "Restart" if needed
5. All changes instant (HTMX)
```

### Share Model Publicly
```
1. Upload with "Enable Public Tunnel" checked
2. Get public URL from success response
3. Share URL with anyone
4. They can predict via public link
5. Monitor usage in logs
```

## 📞 Support

For issues or questions:
1. Check DASHBOARD.md documentation
2. Review error logs via "View Logs"
3. Check test output: `pytest tests/ -v`
4. Verify Docker containers: `docker ps`

Enjoy using PRISM! 🚀

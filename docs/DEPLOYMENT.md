# Deployment Guide

This guide details the recommended procedures for deploying the **AI-Driven Candlestick Prediction Platform** to production environments.

---

## 1. Containerized Deployment (Docker)

Docker is the preferred choice for deploying both the interactive dashboard and the FastAPI web service to any container orchestration engine (e.g. AWS ECS, GCP Cloud Run, Kubernetes).

The repository contains a standard `Dockerfile` that packages the application, locks python dependencies, and runs the dashboard by default.

### Build the Image
```bash
docker build -t ai-candle-predictor:latest .
```

### Deploying the REST API
To run the REST API service inside a container:
```bash
docker run -d \
  --name ai-candle-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ai-candle-predictor:latest \
  uv run uvicorn ai_candle_predictor.presentation.api.main:app --host 0.0.0.0 --port 8000
```
- `-p 8000:8000` binds the REST API to the host port.
- `-v` options ensure persistent volumes are mapped to prevent loss of ingested data and trained model joblib files when the container restarts.

### Deploying the Dashboard
To run the Streamlit dashboard service:
```bash
docker run -d \
  --name ai-candle-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ai-candle-predictor:latest
```

---

## 2. Server Deployment (FastAPI on Cloud VMs)

If deploying to a Virtual Private Server (VPS) or cloud instances (AWS EC2, Google Compute Engine, DigitalOcean), it is best practice to run the application using `uvicorn` managed by a process manager (such as `systemd`) and reverse-proxied behind `Nginx`.

### Systemd Configuration Example (`/etc/systemd/system/ai-candle-api.service`)
```ini
[Unit]
Description=AI Candlestick Predictor FastAPI Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-driven-candle-stick-prediction
ExecStart=/home/ubuntu/.local/bin/uv run uvicorn ai_candle_predictor.presentation.api.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy Config (`/etc/nginx/sites-available/ai-candle`)
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 3. Deploying Streamlit Dashboard to Streamlit Sharing / Cloud

The platform's frontend can be instantly hosted via [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your repository to GitHub.
2. Sign in to Streamlit Community Cloud and click **New App**.
3. Select your repository, the `master` branch, and set the main file path to:
   `src/ai_candle_predictor/presentation/dashboard/app.py`
4. Expand the Advanced Settings and input any necessary Environment Variables (such as custom tickers or configurations).
5. Click **Deploy**. Streamlit Cloud will parse `pyproject.toml` and launch your dashboard automatically.

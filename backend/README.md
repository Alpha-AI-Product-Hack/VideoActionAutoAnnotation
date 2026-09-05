# Backend (FastAPI)

## 安装

在项目根目录执行：

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 启动

```powershell
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

可选环境变量：

- `CORS_ORIGINS`：逗号分隔的允许来源列表，默认 `http://localhost:5173,http://127.0.0.1:5173`
- `LOG_LEVEL`：默认 `INFO`

## API

- `GET /api/health`
- `POST /api/annotate`（multipart/form-data）
  - `video`: 文件
  - `rules`: JSON 字符串（可选）

示例：

```powershell
curl http://127.0.0.1:8000/api/health

curl -X POST http://127.0.0.1:8000/api/annotate `
  -F "video=@clip.mp4" `
  -F "rules={\"actions\":[\"pick_up\"],\"objects\":[\"cup\"],\"min_duration_ms\":250,\"min_confidence\":0.3}"
```

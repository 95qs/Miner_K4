# K4-service – 日志异常检测 REST API

基于 FastAPI + Docker 部署 K4 日志异常检测模型，对外暴露单一 REST 接口。

## 目录结构

```
K4-service/
├── service/                 ← 核心服务代码（Python package）
│   ├── __init__.py
│   ├── api.py              # FastAPI 应用 + 路由
│   ├── config.py           # 配置（设备、路径、超参数）
│   ├── schemas.py          # Pydantic 请求/响应模型
│   ├── engine.py           # K4 推理引擎（封装加载+embedding+PRDC+检测）
│   ├── model_loader.py     # 模型持久化（保存/加载）
│   ├── preprocess.py       # 日志归一化（复用 K4/ 的实现）
│   └── train_service.py    # 训练脚本
├── models/                 ← 训练产出的模型文件（.gitignore，勿提交）
│   └── <model_version>/
│       ├── config.json
│       ├── normal_embeddings.npy
│       ├── scaler.pkl
│       ├── detector.pkl
│       └── threshold.json
├── tests/                  ← 单元测试
├── Dockerfile.gpu          # NVIDIA GPU 构建
├── Dockerfile.cpu         # CPU 构建
├── docker-compose.yml      # GPU 部署
├── docker-compose.cpu.yml  # CPU 部署
├── requirements.txt
├── .env / .env.example
└── README.md
```

## 快速开始

### 前置条件

- **GPU 部署**：NVIDIA CUDA 12.1 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **CPU 部署**：仅需 Docker
- **本地开发**：`pip install -r requirements.txt`，Python 3.10+

### Step 1 – 训练模型（一次性）

```bash
# 从 K4 自带的小规模数据集训练（GPU）
python -m service.train_service \
    --data-path ../K4/syslog_dev \
    --model-version default \
    --embedder all-MiniLM-L6-v2 \
    --detector gmm --k 5

# 或从任意 JSONL 文件训练（CPU）
python -m service.train_service \
    --data-path /path/to/my/logs \
    --model-version my_model \
    --train-file train_normal.jsonl \
    --device cpu
```

训练产物自动保存到 `models/<model_version>/` 目录下。

### Step 2 – 启动服务

**Docker（GPU）**
```bash
docker build -f Dockerfile.gpu -t k4-service:latest .
docker compose up
```

**Docker（CPU）**
```bash
docker build -f Dockerfile.cpu -t k4-service:cpu .
docker compose -f docker-compose.cpu.yml up
```

**本地开发**
```bash
uvicorn service.api:app --reload --host 0.0.0.0 --port 8000
```

### Step 3 – 调用接口

```bash
# 单条检测
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"log": "[799 WARNING][rafale.c:14876]CPU IERR Detected: CPU Error Status Register 0x12 Value = 0xF"}'

# 批量检测
curl -X POST http://localhost:8000/api/v1/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"logs": ["CPU IERR Detected: 0x12", "link status channel0=1 channel1=1", "system startup ok"]}'

# 健康检查
curl http://localhost:8000/health

# 模型信息
curl http://localhost:8000/models/default/info
```

### API 文档

服务启动后访问 http://localhost:8000/docs （Swagger UI）或 http://localhost:8000/redoc（ReDoc）。

## 接口说明

### `POST /api/v1/detect`

单条日志检测。

| 参数 | 类型 | 说明 |
|------|------|------|
| `log` | string | 原始日志文本 |
| `model_version` | string? | 使用哪个模型版本（默认加载 `.env` 中指定的版本）|
| `return_prdc` | bool | 是否在响应中包含 PRDC 四维特征 |
| `return_normalized` | bool | 是否返回归一化后的日志文本 |

**响应字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_fault` | bool | 是否为故障日志 |
| `confidence` | float | 置信度，范围 [0, 1]，1.0 = 高置信度故障，0.0 = 高置信度正常 |
| `raw_score` | float | 检测器原始异常分数 |
| `threshold` | float | 判决阈值 |
| `model_version` | string | 使用的模型版本 |
| `prdc` | object? | PRDC 四维特征 |
| `normalized_log` | string? | 归一化后的日志文本 |

### `POST /api/v1/detect/batch`

批量日志检测，最多 1000 条/请求。

### `GET /health`

健康检查与就绪探针，返回服务状态、加载的模型、GPU 可用性。

### `GET /models/{version}/info`

返回指定模型的元信息（嵌入模型、检测器类型、训练样本数等）。

## 置信度说明

`confidence` 通过 sigmoid 映射将原始分数映射到 [0, 1]：

```
confidence = 1 / (1 + exp(-(score - threshold) / std))
```

- `confidence ≈ 1.0`：模型高度确信这条日志是故障
- `confidence ≈ 0.0`：模型高度确信这条日志是正常
- `confidence ≈ 0.5`：不确定

## GPU / CPU 切换

| 场景 | 操作 |
|------|------|
| 有 NVIDIA GPU | `docker compose up`（默认） |
| 仅 CPU | `docker compose -f docker-compose.cpu.yml up` |
| 修改默认设备 | 编辑 `.env`：`DEVICE=cpu` |

## 部署新模型

训练新模型后，设置 `.env` 中的 `DEFAULT_MODEL_VERSION` 为新版本名，然后重启服务：

```bash
# 训练
python -m service.train_service --data-path /data/v2 --model-version syslog_v2 --embedder all-mpnet-base-v2

# 更新 .env
echo "DEFAULT_MODEL_VERSION=syslog_v2" > .env

# 重启
docker compose restart
```

## 目录不动原有 K4 代码

所有新增代码放在 `K4-service/` 目录下，完全独立于 `K4/`。K4 的核心算法（PRDC 描述子、四种检测器、日志归一化）已完整内嵌到 `service/` 目录中，服务不再依赖原始 K4 文件夹。

## 开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest

# 带覆盖率测试
pytest --cov=service --cov-report=term-missing
```

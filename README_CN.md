# OpenScaner

![OpenScaner 项目横幅](assets/openscaner-banner.png)

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![CPU-only inference](https://img.shields.io/badge/Inference-CPU--only-00A6D6)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white)
![SHA-256 verified models](https://img.shields.io/badge/Models-SHA--256%20verified-4C566A)
[![English README](https://img.shields.io/badge/Docs-EN-2563EB)](README.md)

Vibed coded with Codex.

OpenScaner 是一个面向文档边界检测和扫描校正的开源项目。它把多个 CPU-only 候选模型、传统轮廓算法、后处理策略和本地服务封装在一起，用于从图片中自动定位纸张/文档四角，并输出可用于透视矫正的扫描结果。

项目重点是可复现、可审计和本地部署：模型权重会通过固定文件名、字节大小和 SHA-256 进行校验；服务端使用 FastAPI、SQLite 和本地文件存储；批处理 worker 与 API 共享同一个数据目录。

## 亮点

- CPU-only 推理：面向不依赖 GPU 的本地文档扫描场景。
- 多候选策略：支持 DocAligner、MobileSAM、PP-LCNet、PP-LiteSeg、Fast-SCNN、YOLO11n-seg 和传统轮廓检测等候选来源。
- 自动边界检测：`docaligner_prompted_mobile_sam` 可以仅基于源图像自动生成 MobileSAM prompt，不需要运行时手工 ROI、参考角点或目标坐标。
- 完整性校验：模型、policy 和 manifest 均以固定 SHA-256、大小和文件名绑定，避免加载错误或被替换的权重。
- Web UI 与 API：提供单图同步扫描、批量任务、结果下载和本地 worker。
- 可复现实验：包含训练、校准、评估、benchmark 和第三方许可证说明。

## 快速开始

安装依赖并下载第三方模型权重：

```bash
uv sync --group ml
./scripts/download_models.sh
```

仓库会保留 OpenScaner 自训练的权重；部分第三方预训练权重不直接提交到 Git，而是由下载脚本读取 `models/manifest.json` 后获取，并在使用前校验 SHA-256。

## 启动本地服务

安装依赖：

```bash
uv sync --group ml
```

启动 Web UI 和 API：

```bash
OPENSCANER_AUTH_DISABLED=true \
OPENSCANER_STORAGE_ROOT=/tmp/openscaner-service \
uv run uvicorn openscaner.service.api:create_app --factory --host 127.0.0.1 --port 8000
```

在第二个终端启动批处理 worker：

```bash
OPENSCANER_AUTH_DISABLED=true \
OPENSCANER_STORAGE_ROOT=/tmp/openscaner-service \
uv run python -m openscaner.service.worker
```

然后打开：

```text
http://127.0.0.1:8000/
```

如果启用了认证，可以在页面中输入 API key 或 Web password。

## Docker 部署

```bash
cp .env.example .env
docker compose up --build
```

正式部署前请修改 `.env`：

- 设置 `OPENSCANER_WEB_PASSWORD`
- 设置 `OPENSCANER_API_KEYS`
- 保持 `OPENSCANER_AUTH_DISABLED=false`

## API 示例

同步扫描单张图片：

```bash
curl -H "X-API-Key: change-me-api-key" \
  -F "image=@page.jpg" \
  http://localhost:8000/api/v1/scan
```

创建异步批处理任务：

```bash
curl -H "X-API-Key: change-me-api-key" \
  -F "images=@page1.jpg" -F "images=@page2.jpg" \
  http://localhost:8000/api/v1/jobs > job.json
```

查询任务状态：

```bash
JOB_ID="$(python - <<'PY'
import json
print(json.load(open("job.json"))["id"])
PY
)"

curl -H "X-API-Key: change-me-api-key" \
  "http://localhost:8000/api/v1/jobs/${JOB_ID}"
```

下载批处理结果：

```bash
curl -H "X-API-Key: change-me-api-key" \
  -o results.zip \
  "http://localhost:8000/api/v1/jobs/${JOB_ID}/download.zip"
```

## 常用环境变量

- `OPENSCANER_STORAGE_ROOT`：本地数据目录，默认 `data`
- `OPENSCANER_MODEL_DIR`：模型目录，默认 `models`
- `OPENSCANER_DEFAULT_ADAPTER`：默认候选模型，默认 `docaligner_pp_lcnet_fusion`
- `OPENSCANER_WEB_PASSWORD`：Web UI 密码
- `OPENSCANER_API_KEYS`：逗号分隔的 API key 列表
- `OPENSCANER_RETENTION_DAYS`：任务保留天数，默认 `30`
- `OPENSCANER_KEEP_INPUTS`：是否保留上传原图，默认 `true`
- `OPENSCANER_MAX_UPLOAD_MB`：单文件上传大小限制，默认 `50`
- `OPENSCANER_MAX_BATCH_ITEMS`：单批任务图片数量上限，默认 `500`
- `OPENSCANER_AUTH_DISABLED`：是否关闭认证，默认 `false`

## 项目结构

```text
.
├── models/                         # 模型权重、policy 和 manifest
├── scripts/                        # 模型下载和校验脚本
├── src/openscaner/
│   ├── adapters/                   # 各类边界检测候选模型/算法适配器
│   ├── fusion/                     # 多候选信号融合、policy 和模型身份绑定
│   ├── models/                     # PyTorch 模型结构
│   ├── prompted_sam/               # DocAligner 驱动 MobileSAM prompt 的自动流程
│   ├── refiner/                    # 局部角点细化逻辑
│   ├── service/                    # FastAPI API、Web UI、SQLite store、worker
│   ├── third_party/                # 第三方声明和许可证文本
│   ├── training/                   # 数据、训练、校准和导出流程
│   ├── benchmark.py                # 候选模型 benchmark
│   ├── evaluate.py                 # 评估与结果对比
│   ├── geometry.py                 # 四边形排序、误差和透视校正
│   ├── orientation.py              # 文档方向分类和旋转校正
│   └── postprocess.py              # mask 到文档四边形的后处理
├── tests/                          # 测试
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 模型与许可证

OpenScaner 集成了多个第三方模型和实现思路。相关来源、许可证、权重哈希和使用限制记录在：

- `models/manifest.json`
- `src/openscaner/third_party/`
- `src/openscaner/third_party/licenses/`

如果你计划重新分发模型权重、提供公开在线服务或把本项目用于商业产品，请先核对对应上游项目和权重的许可证要求。

## 开发与测试

安装开发依赖：

```bash
uv sync --group dev --group ml
```

运行测试：

```bash
uv run pytest
```

部分真实模型 smoke test 需要本地模型权重，可按测试标记选择性运行。

## 适用场景

- 本地文档扫描和透视校正
- 文档边界检测模型 benchmark
- 多模型候选融合实验
- CPU-only 推理服务原型
- 可审计模型发布和权重完整性校验流程

## 安全提示

不要把生产环境 `.env` 提交到 Git。部署时请使用强密码和高熵 API key，并保持认证开启。

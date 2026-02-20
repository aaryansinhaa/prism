# PRISM

### Predictive Runtime and Inference Serving Module

> A lightweight, containerized, laptop-native inference serving system for publishing predictive models through a unified public interface.

---

## 1. Motivation

Machine learning practitioners frequently build **toy models, prototypes, or research artifacts** that they want to share publicly.

However:

* Sharing notebooks exposes implementation details.
* Deploying to cloud platforms introduces cost and operational overhead.
* Traditional serving frameworks assume production-scale infrastructure.
* There is no simple “publish model → get link” workflow for local-first users.

**PRISM solves this problem.**

PRISM allows users to:

1. Upload a trained predictive model
2. Automatically isolate it inside a container
3. Expose a unified inference API
4. Generate a public link for external access

The core philosophy:

> Your laptop is your inference server.

---

## 2. Research Inspiration

PRISM is inspired by ideas from modern ML serving and systems research:

### 2.1 Clipper – Low Latency Prediction Serving

Clipper: A Low-Latency Online Prediction Serving System
* [Paper Link](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)
* Unified model abstraction
* Adaptive batching
* Straggler mitigation
* Container isolation

### 2.2 Multi-Model Serving Systems

Managed MLFlow: A Unified Platform for ML Lifecycle

* Model lifecycle management
* Versioning
* Model registry concepts

### 2.3 Serverless / Lightweight Model Deployment

The Case for Learned Index Structures
(Conceptual inspiration for rethinking infrastructure simplicity and modularity.)

### 2.4 High-Performance ML Systems Engineering

Ray: A Distributed Framework for Emerging AI Applications

* Actor-based isolation
* Lightweight distributed abstractions
* Multi-model concurrency patterns

PRISM adapts these ideas for a **local-first, container-native serving platform**.

---

## 3. System Overview

PRISM is a local model serving runtime that:

* Accepts model uploads
* Automatically containerizes them
* Exposes a standardized inference interface
* Generates a public shareable endpoint

---

## 4. Architecture

```
Client
   │
   ▼
Public Link Gateway
   │
   ▼
PRISM Router
   │
   ├── Model Container A
   ├── Model Container B
   └── Model Container C
```

Each model:

* Runs in an isolated container
* Exposes a standardized `/predict` endpoint
* Is managed by a local runtime registry

---

## 5. System Design

### 5.1 Model Upload Flow

1. User uploads:

   * `.pkl` (scikit-learn)
   * `.onnx`
2. PRISM validates format
3. PRISM generates container spec
4. Container is launched
5. Model is registered in the runtime registry
6. Public link is generated

---

### 5.2 Container Isolation

Each model runs in its own container to ensure:

* Dependency isolation
* Fault isolation
* Resource boundaries
* Hot-swappable deployment

Containers expose:

```
POST /predict
```

PRISM communicates internally via RPC or HTTP.

---

### 5.3 Unified Model Interface

All models must implement:

```python
class BaseModel:
    def predict(self, input):
        ...
```

Supported in v1:

* Scikit-learn
* ONNX Runtime

Future versions may support:

* TorchScript
* TensorFlow SavedModel

---

## 6. Core System Features

### 6.1 Request Batching

PRISM implements dynamic batching:

* Accumulates requests in a queue
* Triggers inference when:

  * Batch size threshold is reached
  * Latency deadline expires

Benefits:

* Improved throughput
* Better CPU utilization
* Reduced per-request overhead

---

### 6.2 Prediction Caching

PRISM supports:

* In-memory LRU caching
* Optional Redis-backed cache

This reduces repeated computation for:

* Deterministic models
* Repeated evaluation scenarios

---

### 6.3 Multi-Modal Input Support

PRISM supports:

* JSON structured input
* Numeric arrays
* Text input (for NLP models)
* Image tensors (base64 or binary)

Input normalization layer ensures:

* Format validation
* Schema enforcement
* Type conversion

---

### 6.4 Multi-Model Hosting

Users can host:

* Multiple models
* Multiple versions
* Different frameworks

All models share:

* One public gateway
* Unified API contract
* Independent containers

---

## 7. Public Link Generation

PRISM provides:

```
https://<user-ip>:<port>/model/<model-id>/predict
```

Future support:

* Reverse proxy tunneling (ngrok-style)
* Authentication tokens
* Rate limiting

---

## 8. Performance Goals (v1)

Target metrics:

* P50 latency < 20ms (local inference)
* Adaptive batching support
* Multi-model concurrency
* Graceful degradation under load

---

## 9. Technical Stack

* Python
* FastAPI
* ONNX Runtime
* Scikit-learn
* Docker
* Poetry (dependency management)
* Uvicorn (ASGI server)

---

## 10. Limitations (v1)

* Only predictive models supported
* No distributed scaling
* No autoscaling
* No GPU scheduling
* Local-only runtime

---

## 11. Why PRISM is Interesting (From a Systems Perspective)

PRISM explores:

* Model serving abstraction layers
* Container-based isolation
* Latency-aware batching
* Multi-tenant inference on edge devices
* Laptop-native model deployment

Unlike cloud serving systems, PRISM focuses on:

> Local-first inference publishing with minimal operational overhead.

---

## 12. Future Work

* Model versioning
* A/B testing
* Model ensembling
* Straggler mitigation
* Resource-aware scheduling
* GPU-aware batching
* Horizontal scaling cluster mode
* WASM sandbox runtime

---

## 13. Example Usage

Upload model:

```
prism upload churn_model.pkl
```

Server response:

```
Model deployed successfully.
Public endpoint:
http://192.168.0.14:8000/model/churn/predict
```

Inference request:

```bash
curl -X POST \
  http://192.168.0.14:8000/model/churn/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [ ... ]}'
```

---

## 14. Benchmarks

Baseline (bare-metal) latency and throughput numbers, plus reproduction steps, now live in `BENCHMARKS.md`. Treat those as source-of-truth when evaluating future runtime overhead.

---

# Closing Vision

PRISM aims to make model deployment as simple as:

> Train → Upload → Share Link

No cloud.
No notebook exposure.
No infrastructure complexity.

Just a predictive runtime layer sitting on your own machine.

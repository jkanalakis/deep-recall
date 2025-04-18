# Deep Recall Monitoring

This directory contains configurations for monitoring the Deep Recall system using Prometheus and Grafana.

## Overview

The monitoring stack consists of:

- **Prometheus**: Collects and stores metrics from all services
- **Grafana**: Visualizes metrics in customizable dashboards
- **Node Exporter**: Collects system-level metrics (CPU, memory, disk, etc.)
- **cAdvisor**: Collects container metrics

## Directory Structure

- `prometheus.yml`: Prometheus configuration file with scrape configs for all services
- `grafana/`: Grafana configuration and dashboards
  - `provisioning/datasources/`: Prometheus datasource configuration
  - `provisioning/dashboards/`: Dashboard provisioning configuration
  - `dashboards/`: Pre-configured Grafana dashboards in JSON format

## Available Dashboards

1. **System Metrics Dashboard**: General system metrics including memory usage, CPU usage, and HTTP request metrics
2. **Model Performance Dashboard**: Metrics specific to model performance including inference time, request rate, and GPU utilization
3. **Memory Service Dashboard**: Metrics specific to the memory service including request rate, latency, and database metrics

## Deployment

### Docker Compose

The monitoring services are included in the `docker-compose.dev.yaml` file and can be started along with other services:

```bash
cd deployments/docker
docker-compose -f docker-compose.dev.yaml up -d
```

### Kubernetes

The Prometheus and Grafana services are defined in the Kubernetes manifests:

```bash
# Apply using kustomize
kubectl apply -k deployments/kubernetes/base
```

## Accessing Dashboards

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` 
  - Default credentials: admin/admin

## Adding Custom Metrics

1. Modify your service to expose Prometheus metrics
2. Update the Prometheus configuration to scrape your new metrics endpoint
3. Create or update Grafana dashboards to visualize the new metrics

## Service Instrumentation

Each service in the Deep Recall system exposes metrics on a dedicated metrics endpoint:

- **Inference Service**: Metrics exposed on port 8001
- **Memory Service**: Metrics exposed on port 8001 
- **Orchestrator**: Metrics exposed on port 8001

## Alerting

Alerting rules can be added to the Prometheus configuration to notify when metrics exceed certain thresholds. Alerts can be sent via Alertmanager to various channels such as email, Slack, or PagerDuty. 
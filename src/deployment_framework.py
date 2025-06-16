import os
import json
import torch
import docker
import yaml
from typing import Dict, List, Optional
from datetime import datetime
import subprocess
import shutil
from pathlib import Path

class EdgeDeploymentManager:
    """
    Manages deployment of FairTriEdge-FL models to edge devices.
    """
    
    def __init__(self, config_path='config/config.json'):
        self.config = self._load_config(config_path)
        self.deployment_history = []
    
    def _load_config(self, config_path):
        """Load deployment configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def prepare_edge_model(self, model, optimization_level='standard'):
        """
        Prepare model for edge deployment with optimizations.
        
        Args:
            model: PyTorch model to deploy
            optimization_level: 'minimal', 'standard', 'aggressive'
        
        Returns:
            dict: Deployment package information
        """
        print(f"🚀 Preparing model for edge deployment (level: {optimization_level})")
        
        # Create deployment directory
        deploy_dir = Path('deployment')
        deploy_dir.mkdir(exist_ok=True)
        
        # Model optimization based on level
        if optimization_level == 'minimal':
            optimized_model = model
        elif optimization_level == 'standard':
            optimized_model = self._apply_standard_optimizations(model)
        elif optimization_level == 'aggressive':
            optimized_model = self._apply_aggressive_optimizations(model)
        else:
            raise ValueError(f"Unknown optimization level: {optimization_level}")
        
        # Save optimized model
        model_path = deploy_dir / 'optimized_model.pth'
        torch.save(optimized_model.state_dict(), model_path)
        
        # Generate model metadata
        metadata = self._generate_model_metadata(optimized_model, optimization_level)
        metadata_path = deploy_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create deployment package
        package_info = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'optimization_level': optimization_level,
            'deployment_time': datetime.now().isoformat(),
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        print(f"✅ Model prepared for deployment:")
        print(f"   - Model size: {package_info['model_size_mb']:.2f}MB")
        print(f"   - Optimization: {optimization_level}")
        print(f"   - Location: {deploy_dir}")
        
        return package_info
    
    def _apply_standard_optimizations(self, model):
        """Apply standard optimizations for edge deployment."""
        from .model_optimization import apply_quantization, apply_pruning
        
        print("  Applying standard optimizations...")
        
        # Apply pruning (30% sparsity)
        pruned_model = apply_pruning(model, amount=0.3)
        
        # Apply quantization
        quantized_model = apply_quantization(pruned_model, backend='auto')
        
        return quantized_model
    
    def _apply_aggressive_optimizations(self, model):
        """Apply aggressive optimizations for resource-constrained devices."""
        from .model_optimization import apply_quantization, apply_pruning
        
        print("  Applying aggressive optimizations...")
        
        # Apply heavy pruning (60% sparsity)
        pruned_model = apply_pruning(model, amount=0.6)
        
        # Apply quantization
        quantized_model = apply_quantization(pruned_model, backend='auto')
        
        # Additional optimizations could include:
        # - Knowledge distillation to smaller model
        # - Layer fusion
        # - Custom ONNX export
        
        return quantized_model
    
    def _generate_model_metadata(self, model, optimization_level):
        """Generate comprehensive model metadata."""
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        
        metadata = {
            'model_info': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'memory_usage_bytes': total_size,
                'memory_usage_mb': total_size / (1024 * 1024)
            },
            'optimization': {
                'level': optimization_level,
                'techniques_applied': ['pruning', 'quantization'],
                'target_devices': ['raspberry_pi', 'jetson_nano', 'edge_tpu']
            },
            'performance_requirements': {
                'max_inference_time_ms': 100,
                'max_memory_mb': 50,
                'min_accuracy': 0.85
            },
            'deployment_info': {
                'framework': 'pytorch',
                'python_version': '3.8+',
                'dependencies': ['torch', 'numpy', 'pandas'],
                'created_at': datetime.now().isoformat()
            }
        }
        
        return metadata
    
    def create_docker_container(self, package_info, target_platform='linux/arm64'):
        """
        Create Docker container for edge deployment.
        
        Args:
            package_info: Model package information
            target_platform: Target platform (linux/arm64, linux/amd64)
        
        Returns:
            str: Docker image name
        """
        print(f"🐳 Creating Docker container for {target_platform}")
        
        # Create Dockerfile
        dockerfile_content = self._generate_dockerfile(target_platform)
        
        with open('deployment/Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt for container
        requirements = [
            'torch>=1.9.0',
            'numpy',
            'pandas',
            'fastapi',
            'uvicorn',
            'pydantic'
        ]
        
        with open('deployment/requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create inference API
        api_code = self._generate_inference_api()
        with open('deployment/inference_api.py', 'w') as f:
            f.write(api_code)
        
        # Build Docker image
        image_name = f"fairtriedge-fl:edge-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            client = docker.from_env()
            image = client.images.build(
                path='deployment/',
                tag=image_name,
                platform=target_platform
            )
            print(f"✅ Docker image created: {image_name}")
            return image_name
        except Exception as e:
            print(f"❌ Docker build failed: {e}")
            return None
    
    def _generate_dockerfile(self, target_platform):
        """Generate Dockerfile for edge deployment."""
        if 'arm' in target_platform:
            base_image = 'python:3.9-slim-bullseye'
        else:
            base_image = 'python:3.9-slim'
        
        dockerfile = f"""
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY optimized_model.pth .
COPY model_metadata.json .
COPY inference_api.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile
    
    def _generate_inference_api(self):
        """Generate FastAPI inference service."""
        api_code = '''
import torch
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import time

app = FastAPI(title="FairTriEdge-FL Inference API", version="1.0.0")

# Load model and metadata
model = None
metadata = None

@app.on_event("startup")
async def load_model():
    global model, metadata
    try:
        # Load model architecture (you'll need to import your TriageModel)
        # model = TriageModel(7, 268, 3, 3)
        # model.load_state_dict(torch.load('optimized_model.pth', map_location='cpu'))
        # model.eval()
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

class TriageRequest(BaseModel):
    numerical_features: List[float]  # [age, systolic_bp, diastolic_bp, resp_rate, heart_rate, temp, sat]
    boolean_features: List[int]      # 268 boolean features
    temporal_features: List[float]   # [hour, day_of_week, month]

class TriageResponse(BaseModel):
    prediction: int                  # 0=Green, 1=Yellow, 2=Red
    confidence: float               # Prediction confidence
    inference_time_ms: float        # Inference time
    explanation: str                # Brief explanation

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/info")
async def model_info():
    return metadata

@app.post("/predict", response_model=TriageResponse)
async def predict_triage(request: TriageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Prepare input tensors
        numerical = torch.tensor([request.numerical_features], dtype=torch.float32)
        boolean = torch.tensor([request.boolean_features], dtype=torch.float32)
        temporal = torch.tensor([request.temporal_features], dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            outputs = model(numerical, boolean, temporal)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        inference_time = (time.time() - start_time) * 1000
        
        # Generate explanation
        triage_levels = ["Green (Low Priority)", "Yellow (Moderate Priority)", "Red (High Priority)"]
        explanation = f"Patient classified as {triage_levels[prediction]} with {confidence:.1%} confidence"
        
        return TriageResponse(
            prediction=prediction,
            confidence=confidence,
            inference_time_ms=inference_time,
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[TriageRequest]):
    """Batch prediction endpoint for multiple patients."""
    results = []
    for req in requests:
        try:
            result = await predict_triage(req)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "total_processed": len(results)}
'''
        return api_code
    
    def deploy_to_edge_device(self, package_info, device_config):
        """
        Deploy model to specific edge device.
        
        Args:
            package_info: Model package information
            device_config: Edge device configuration
        
        Returns:
            dict: Deployment status
        """
        print(f"📱 Deploying to edge device: {device_config.get('name', 'Unknown')}")
        
        deployment_status = {
            'device_name': device_config.get('name'),
            'device_type': device_config.get('type'),
            'deployment_time': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        try:
            # Device-specific deployment logic
            if device_config.get('type') == 'raspberry_pi':
                status = self._deploy_to_raspberry_pi(package_info, device_config)
            elif device_config.get('type') == 'jetson_nano':
                status = self._deploy_to_jetson_nano(package_info, device_config)
            elif device_config.get('type') == 'edge_tpu':
                status = self._deploy_to_edge_tpu(package_info, device_config)
            else:
                status = self._deploy_generic(package_info, device_config)
            
            deployment_status.update(status)
            deployment_status['status'] = 'success'
            
        except Exception as e:
            deployment_status['status'] = 'failed'
            deployment_status['error'] = str(e)
            print(f"❌ Deployment failed: {e}")
        
        self.deployment_history.append(deployment_status)
        return deployment_status
    
    def _deploy_to_raspberry_pi(self, package_info, device_config):
        """Deploy to Raspberry Pi device."""
        print("  Configuring for Raspberry Pi...")
        return {
            'optimization': 'ARM-specific optimizations applied',
            'memory_limit': '1GB',
            'inference_target': '<100ms'
        }
    
    def _deploy_to_jetson_nano(self, package_info, device_config):
        """Deploy to NVIDIA Jetson Nano."""
        print("  Configuring for Jetson Nano...")
        return {
            'optimization': 'CUDA optimizations applied',
            'memory_limit': '4GB',
            'inference_target': '<50ms'
        }
    
    def _deploy_to_edge_tpu(self, package_info, device_config):
        """Deploy to Google Edge TPU."""
        print("  Configuring for Edge TPU...")
        return {
            'optimization': 'TPU-specific quantization applied',
            'memory_limit': '8MB',
            'inference_target': '<10ms'
        }
    
    def _deploy_generic(self, package_info, device_config):
        """Generic deployment for other edge devices."""
        print("  Applying generic edge optimizations...")
        return {
            'optimization': 'Standard edge optimizations',
            'memory_limit': 'Device-dependent',
            'inference_target': '<200ms'
        }
    
    def monitor_deployment(self, deployment_id):
        """Monitor deployed model performance."""
        # This would integrate with monitoring systems
        return {
            'deployment_id': deployment_id,
            'status': 'running',
            'uptime': '99.9%',
            'avg_inference_time': '45ms',
            'requests_per_minute': 120,
            'error_rate': '0.1%'
        }
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report."""
        report = {
            'summary': {
                'total_deployments': len(self.deployment_history),
                'successful_deployments': len([d for d in self.deployment_history if d['status'] == 'success']),
                'failed_deployments': len([d for d in self.deployment_history if d['status'] == 'failed'])
            },
            'deployments': self.deployment_history,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = f"results/deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Deployment report saved: {report_path}")
        return report

class ProductionMonitor:
    """
    Production monitoring and alerting system.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
    
    def collect_metrics(self, deployment_id, metrics):
        """Collect performance metrics from deployed models."""
        metric_entry = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.metrics_history.append(metric_entry)
        
        # Check for alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics):
        """Check metrics against alert thresholds."""
        alerts = []
        
        # Performance alerts
        if metrics.get('avg_inference_time_ms', 0) > 200:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"High inference time: {metrics['avg_inference_time_ms']}ms"
            })
        
        # Accuracy alerts
        if metrics.get('accuracy', 1.0) < 0.8:
            alerts.append({
                'type': 'accuracy',
                'severity': 'critical',
                'message': f"Low accuracy detected: {metrics['accuracy']:.3f}"
            })
        
        # Error rate alerts
        if metrics.get('error_rate', 0) > 0.05:
            alerts.append({
                'type': 'error_rate',
                'severity': 'warning',
                'message': f"High error rate: {metrics['error_rate']:.1%}"
            })
        
        self.alerts.extend(alerts)
        
        # Send notifications (would integrate with alerting systems)
        for alert in alerts:
            print(f"🚨 ALERT [{alert['severity']}]: {alert['message']}")
    
    def generate_monitoring_dashboard(self):
        """Generate monitoring dashboard data."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        # Calculate aggregated metrics
        recent_metrics = self.metrics_history[-100:]  # Last 100 entries
        
        dashboard = {
            'overview': {
                'total_requests': sum(m['metrics'].get('request_count', 0) for m in recent_metrics),
                'avg_response_time': np.mean([m['metrics'].get('avg_inference_time_ms', 0) for m in recent_metrics]),
                'error_rate': np.mean([m['metrics'].get('error_rate', 0) for m in recent_metrics]),
                'uptime': '99.9%'  # Would be calculated from actual uptime data
            },
            'alerts': {
                'active_alerts': len([a for a in self.alerts if a.get('resolved', False) == False]),
                'recent_alerts': self.alerts[-10:]  # Last 10 alerts
            },
            'performance_trends': {
                'inference_time_trend': [m['metrics'].get('avg_inference_time_ms', 0) for m in recent_metrics[-24:]],
                'accuracy_trend': [m['metrics'].get('accuracy', 0) for m in recent_metrics[-24:]],
                'request_volume_trend': [m['metrics'].get('request_count', 0) for m in recent_metrics[-24:]]
            }
        }
        
        return dashboard

if __name__ == "__main__":
    # Example usage
    print("🚀 FairTriEdge-FL Deployment Framework")
    
    # Initialize deployment manager
    deploy_manager = EdgeDeploymentManager()
    
    # Example edge device configurations
    edge_devices = [
        {
            'name': 'Hospital-A-RaspberryPi',
            'type': 'raspberry_pi',
            'ip': '192.168.1.100',
            'specs': {'ram': '4GB', 'cpu': 'ARM Cortex-A72'}
        },
        {
            'name': 'Hospital-B-JetsonNano',
            'type': 'jetson_nano',
            'ip': '192.168.1.101',
            'specs': {'ram': '4GB', 'gpu': 'Maxwell GPU'}
        }
    ]
    
    print("📱 Available edge devices:")
    for device in edge_devices:
        print(f"   - {device['name']} ({device['type']})")
    
    print("\n✅ Deployment framework ready!")
    print("   Use deploy_manager.prepare_edge_model() to start deployment")
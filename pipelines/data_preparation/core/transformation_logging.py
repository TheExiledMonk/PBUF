"""
Transformation logging and audit trail system for the data preparation framework.

This module provides detailed logging of all transformation steps with formula
references, processing summaries suitable for publication materials, and
performance metrics tracking.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import numpy as np

from .schema import StandardDataset


@dataclass
class TransformationStep:
    """Detailed information about a single transformation step."""
    step_name: str
    formula: Optional[str] = None
    description: Optional[str] = None
    input_description: Optional[str] = None
    output_description: Optional[str] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    parameters: Optional[Dict[str, Any]] = None
    references: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    timestamp: Optional[str] = None
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationStep':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProcessingSummary:
    """Complete processing summary for a dataset."""
    dataset_name: str
    dataset_type: str
    processing_start: str
    processing_end: str
    total_duration: float
    transformation_steps: List[TransformationStep]
    validation_results: Dict[str, Any]
    input_metadata: Dict[str, Any]
    output_metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert TransformationStep objects to dicts
        data['transformation_steps'] = [step.to_dict() for step in self.transformation_steps]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingSummary':
        """Create from dictionary."""
        # Convert step dicts back to TransformationStep objects
        data['transformation_steps'] = [
            TransformationStep.from_dict(step) for step in data['transformation_steps']
        ]
        return cls(**data)


class TransformationLogger:
    """
    Comprehensive transformation logging system.
    
    Tracks all transformation steps with detailed metadata, formulas,
    and performance metrics for complete audit trails and publication
    materials.
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        output_directory: Optional[Path] = None
    ):
        """
        Initialize transformation logger for a dataset.
        
        Args:
            dataset_name: Name of dataset being processed
            dataset_type: Type of dataset (sn, bao, cmb, etc.)
            output_directory: Directory for log outputs
        """
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.output_directory = output_directory or Path("logs/transformations")
        try:
            if not self.output_directory.exists():
                self.output_directory.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError, TypeError):
            # Directory already exists, other OS error, or mocking interference - continue
            pass
        
        # Processing tracking
        self.processing_start_time: Optional[datetime] = None
        self.processing_end_time: Optional[datetime] = None
        
        # Transformation steps
        self.transformation_steps: List[TransformationStep] = []
        
        # Current step tracking
        self.current_step: Optional[TransformationStep] = None
        self.step_start_time: Optional[float] = None
        
        # Metadata tracking
        self.input_metadata: Dict[str, Any] = {}
        self.output_metadata: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {}
        
        # Environment information
        self.environment_info = self._collect_environment_info()
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for reproducibility."""
        import sys
        import platform
        
        try:
            import psutil
            memory_info = {
                "total_memory": psutil.virtual_memory().total,
                "available_memory": psutil.virtual_memory().available
            }
        except ImportError:
            memory_info = {"memory_info": "psutil not available"}
        
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **memory_info
        }
        
        # Try to get git commit if available
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent.parent.parent
            ).decode().strip()
            env_info["git_commit"] = git_commit
        except:
            env_info["git_commit"] = "unknown"
        
        return env_info
    
    def start_processing(self, input_metadata: Dict[str, Any] = None):
        """Mark the start of processing."""
        self.processing_start_time = datetime.now(timezone.utc)
        self.input_metadata = input_metadata or {}
        
        print(f"Starting transformation logging for dataset: {self.dataset_name}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Processing started at: {self.processing_start_time.isoformat()}")
    
    def end_processing(self, output_metadata: Dict[str, Any] = None):
        """Mark the end of processing."""
        self.processing_end_time = datetime.now(timezone.utc)
        self.output_metadata = output_metadata or {}
        
        if self.processing_start_time:
            duration = (self.processing_end_time - self.processing_start_time).total_seconds()
            print(f"Processing completed in {duration:.2f} seconds")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
    
    @contextmanager
    def log_transformation_step(
        self,
        step_name: str,
        formula: Optional[str] = None,
        description: Optional[str] = None,
        input_description: Optional[str] = None,
        output_description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        references: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None
    ):
        """
        Context manager for logging transformation steps.
        
        Args:
            step_name: Name of the transformation step
            formula: Mathematical formula used (LaTeX format)
            description: Detailed description of the transformation
            input_description: Description of input data
            output_description: Description of output data
            parameters: Parameters used in transformation
            references: Literature references
            assumptions: Physical or mathematical assumptions
        """
        # Create transformation step
        step = TransformationStep(
            step_name=step_name,
            formula=formula,
            description=description,
            input_description=input_description,
            output_description=output_description,
            parameters=parameters,
            references=references,
            assumptions=assumptions,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.current_step = step
        self.step_start_time = time.time()
        
        print(f"  Starting transformation step: {step_name}")
        if formula:
            print(f"    Formula: {formula}")
        if description:
            print(f"    Description: {description}")
        
        try:
            yield step
        finally:
            # Calculate duration
            if self.step_start_time:
                step.duration = time.time() - self.step_start_time
                print(f"    Completed in {step.duration:.3f} seconds")
            
            # Add to transformation steps
            self.transformation_steps.append(step)
            self.current_step = None
            self.step_start_time = None
    
    def log_data_shape(self, data: Union[np.ndarray, StandardDataset], label: str = "data"):
        """Log the shape of data arrays."""
        if self.current_step is None:
            return
        
        if isinstance(data, StandardDataset):
            shapes = {
                "z": data.z.shape if hasattr(data, 'z') and data.z is not None else None,
                "observable": data.observable.shape if hasattr(data, 'observable') and data.observable is not None else None,
                "uncertainty": data.uncertainty.shape if hasattr(data, 'uncertainty') and data.uncertainty is not None else None,
                "covariance": data.covariance.shape if hasattr(data, 'covariance') and data.covariance is not None else None
            }
            print(f"    {label} shapes: {shapes}")
            
            if label.lower().startswith('input'):
                self.current_step.input_shape = shapes
            elif label.lower().startswith('output'):
                self.current_step.output_shape = shapes
                
        elif isinstance(data, np.ndarray):
            shape = data.shape
            print(f"    {label} shape: {shape}")
            
            if label.lower().startswith('input'):
                self.current_step.input_shape = shape
            elif label.lower().startswith('output'):
                self.current_step.output_shape = shape
    
    def log_parameter(self, name: str, value: Any, description: Optional[str] = None):
        """Log a parameter used in the current transformation step."""
        if self.current_step is None:
            return
        
        if self.current_step.parameters is None:
            self.current_step.parameters = {}
        
        self.current_step.parameters[name] = {
            "value": value,
            "description": description
        }
        
        print(f"    Parameter {name}: {value}")
        if description:
            print(f"      {description}")
    
    def log_validation_results(self, validation_results: Dict[str, Any]):
        """Log validation results."""
        self.validation_results = validation_results
        
        print("Validation Results:")
        for validation_type, result in validation_results.items():
            # Handle both dict and non-dict results
            if isinstance(result, dict):
                status = "PASSED" if result.get('passed', False) else "FAILED"
                print(f"  {validation_type}: {status}")
                if 'details' in result:
                    for key, value in result['details'].items():
                        print(f"    {key}: {value}")
            else:
                # Handle non-dict results (strings, booleans, etc.)
                print(f"  {validation_type}: {result}")
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from transformation steps."""
        if not self.transformation_steps:
            return
        
        # Step-level metrics
        step_durations = [step.duration for step in self.transformation_steps if step.duration]
        
        if step_durations:
            self.performance_metrics = {
                "total_steps": len(self.transformation_steps),
                "total_transformation_time": sum(step_durations),
                "average_step_time": sum(step_durations) / len(step_durations),
                "longest_step": max(step_durations),
                "shortest_step": min(step_durations),
                "step_breakdown": {
                    step.step_name: step.duration 
                    for step in self.transformation_steps 
                    if step.duration
                }
            }
        
        # Overall processing metrics
        if self.processing_start_time and self.processing_end_time:
            total_duration = (self.processing_end_time - self.processing_start_time).total_seconds()
            transformation_time = sum(step_durations) if step_durations else 0
            overhead_time = total_duration - transformation_time
            
            self.performance_metrics.update({
                "total_processing_time": total_duration,
                "transformation_overhead": overhead_time,
                "transformation_efficiency": (transformation_time / total_duration * 100) if total_duration > 0 else 0
            })
    
    def generate_processing_summary(self) -> ProcessingSummary:
        """Generate complete processing summary."""
        if not self.processing_start_time:
            raise ValueError("Processing not started - call start_processing() first")
        
        if not self.processing_end_time:
            self.end_processing()
        
        total_duration = (self.processing_end_time - self.processing_start_time).total_seconds()
        
        return ProcessingSummary(
            dataset_name=self.dataset_name,
            dataset_type=self.dataset_type,
            processing_start=self.processing_start_time.isoformat(),
            processing_end=self.processing_end_time.isoformat(),
            total_duration=total_duration,
            transformation_steps=self.transformation_steps,
            validation_results=self.validation_results,
            input_metadata=self.input_metadata,
            output_metadata=self.output_metadata,
            environment_info=self.environment_info,
            performance_metrics=self.performance_metrics
        )
    
    def save_processing_summary(self, filename: Optional[str] = None) -> Path:
        """Save processing summary to file."""
        summary = self.generate_processing_summary()
        
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_processing_summary_{timestamp}.json"
        
        output_path = self.output_directory / filename
        
        # Ensure directory exists
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError, TypeError):
            # Directory already exists, other OS error, or mocking interference - continue
            pass
        
        with open(output_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        print(f"Processing summary saved to: {output_path}")
        return output_path
    
    def generate_publication_summary(self) -> Dict[str, Any]:
        """
        Generate processing summary suitable for publication materials.
        
        Returns:
            Dictionary with publication-ready processing information
        """
        summary = self.generate_processing_summary()
        
        # Extract key information for publication
        pub_summary = {
            "dataset_name": summary.dataset_name,
            "dataset_type": summary.dataset_type,
            "processing_date": summary.processing_start.split('T')[0],
            "total_processing_time": f"{summary.total_duration:.2f} seconds",
            "transformation_steps": []
        }
        
        # Summarize transformation steps for publication
        for step in summary.transformation_steps:
            step_info = {
                "step": step.step_name,
                "description": step.description or step.step_name
            }
            
            if step.formula:
                step_info["formula"] = step.formula
            
            if step.references:
                step_info["references"] = step.references
            
            if step.assumptions:
                step_info["assumptions"] = step.assumptions
            
            pub_summary["transformation_steps"].append(step_info)
        
        # Add validation summary
        if summary.validation_results:
            validation_summary = {}
            for validation_type, result in summary.validation_results.items():
                # Handle both dict and non-dict results
                if isinstance(result, dict):
                    validation_summary[validation_type] = "passed" if result.get('passed', False) else "failed"
                else:
                    # Handle non-dict results (strings, booleans, etc.)
                    validation_summary[validation_type] = str(result)
            pub_summary["validation_results"] = validation_summary
        
        # Add performance summary
        if summary.performance_metrics:
            pub_summary["performance"] = {
                "total_steps": summary.performance_metrics.get("total_steps", 0),
                "processing_efficiency": f"{summary.performance_metrics.get('transformation_efficiency', 0):.1f}%"
            }
        
        # Add environment summary
        pub_summary["processing_environment"] = {
            "python_version": summary.environment_info.get("python_version", "unknown").split()[0],
            "platform": summary.environment_info.get("platform", "unknown"),
            "framework_version": "1.0.0"
        }
        
        return pub_summary
    
    def save_publication_summary(self, filename: Optional[str] = None) -> Path:
        """Save publication summary to file."""
        pub_summary = self.generate_publication_summary()
        
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_publication_summary_{timestamp}.json"
        
        output_path = self.output_directory / filename
        
        # Ensure directory exists
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError, TypeError):
            # Directory already exists, other OS error, or mocking interference - continue
            pass
        
        with open(output_path, 'w') as f:
            json.dump(pub_summary, f, indent=2)
        
        print(f"Publication summary saved to: {output_path}")
        return output_path
    
    def generate_latex_summary(self) -> str:
        """
        Generate LaTeX-formatted processing summary for papers.
        
        Returns:
            LaTeX-formatted string describing the processing
        """
        summary = self.generate_processing_summary()
        
        latex_lines = []
        latex_lines.append(f"% Processing summary for {summary.dataset_name}")
        latex_lines.append(f"% Generated on {summary.processing_start.split('T')[0]}")
        latex_lines.append("")
        
        # Dataset processing section
        latex_lines.append("\\subsection{Data Processing}")
        latex_lines.append(f"The {summary.dataset_type.upper()} dataset \\texttt{{{summary.dataset_name}}} ")
        latex_lines.append(f"was processed through {len(summary.transformation_steps)} transformation steps:")
        latex_lines.append("")
        
        # Transformation steps
        latex_lines.append("\\begin{enumerate}")
        for i, step in enumerate(summary.transformation_steps, 1):
            latex_lines.append(f"  \\item \\textbf{{{step.step_name}}}")
            
            if step.description:
                latex_lines.append(f"    {step.description}")
            
            if step.formula:
                latex_lines.append(f"    \\begin{{equation}}")
                latex_lines.append(f"      {step.formula}")
                latex_lines.append(f"    \\end{{equation}}")
            
            if step.assumptions:
                latex_lines.append(f"    Assumptions: {'; '.join(step.assumptions)}")
            
            if step.references:
                latex_lines.append(f"    References: {', '.join(step.references)}")
            
            latex_lines.append("")
        
        latex_lines.append("\\end{enumerate}")
        latex_lines.append("")
        
        # Validation results
        if summary.validation_results:
            latex_lines.append("\\subsection{Validation Results}")
            latex_lines.append("The processed dataset passed the following validation checks:")
            latex_lines.append("\\begin{itemize}")
            
            for validation_type, result in summary.validation_results.items():
                status = "passed" if result.get('passed', False) else "failed"
                latex_lines.append(f"  \\item {validation_type.replace('_', ' ').title()}: {status}")
            
            latex_lines.append("\\end{itemize}")
            latex_lines.append("")
        
        # Performance summary
        if summary.performance_metrics:
            processing_time = summary.performance_metrics.get('total_processing_time', 0)
            latex_lines.append(f"Processing completed in {processing_time:.2f} seconds ")
            latex_lines.append(f"with {summary.performance_metrics.get('transformation_efficiency', 0):.1f}\\% efficiency.")
        
        return "\n".join(latex_lines)
    
    def save_latex_summary(self, filename: Optional[str] = None) -> Path:
        """Save LaTeX summary to file."""
        latex_content = self.generate_latex_summary()
        
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_latex_summary_{timestamp}.tex"
        
        output_path = self.output_directory / filename
        
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX summary saved to: {output_path}")
        return output_path


class SystemHealthMonitor:
    """
    System health monitoring for the data preparation framework.
    
    Tracks system resources, processing performance, and identifies
    potential bottlenecks or issues.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize system health monitor.
        
        Args:
            monitoring_interval: Interval between health checks in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.health_data: List[Dict[str, Any]] = []
        self.monitoring_active = False
        
        # Performance thresholds
        self.thresholds = {
            "memory_usage_percent": 90.0,
            "disk_usage_percent": 95.0,
            "cpu_usage_percent": 95.0,
            "processing_time_seconds": 300.0  # 5 minutes
        }
    
    def start_monitoring(self):
        """Start system health monitoring."""
        self.monitoring_active = True
        self.health_data.clear()
        print("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self.monitoring_active = False
        print("System health monitoring stopped")
    
    def record_health_snapshot(self) -> Dict[str, Any]:
        """Record current system health snapshot."""
        try:
            import psutil
            
            # System metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                }
            }
            
            # Check for threshold violations
            violations = []
            if memory.percent > self.thresholds["memory_usage_percent"]:
                violations.append(f"High memory usage: {memory.percent:.1f}%")
            
            if snapshot["disk"]["percent"] > self.thresholds["disk_usage_percent"]:
                violations.append(f"High disk usage: {snapshot['disk']['percent']:.1f}%")
            
            if cpu_percent > self.thresholds["cpu_usage_percent"]:
                violations.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            snapshot["threshold_violations"] = violations
            
            if self.monitoring_active:
                self.health_data.append(snapshot)
            
            return snapshot
            
        except ImportError:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "psutil not available for system monitoring"
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health during monitoring period."""
        if not self.health_data:
            return {"error": "No health data available"}
        
        # Calculate averages and peaks
        memory_usage = [snapshot["memory"]["percent"] for snapshot in self.health_data if "memory" in snapshot]
        cpu_usage = [snapshot["cpu"]["percent"] for snapshot in self.health_data if "cpu" in snapshot]
        disk_usage = [snapshot["disk"]["percent"] for snapshot in self.health_data if "disk" in snapshot]
        
        # Collect all violations
        all_violations = []
        for snapshot in self.health_data:
            all_violations.extend(snapshot.get("threshold_violations", []))
        
        summary = {
            "monitoring_period": {
                "start": self.health_data[0]["timestamp"],
                "end": self.health_data[-1]["timestamp"],
                "snapshots": len(self.health_data)
            },
            "memory_usage": {
                "average": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "peak": max(memory_usage) if memory_usage else 0,
                "minimum": min(memory_usage) if memory_usage else 0
            },
            "cpu_usage": {
                "average": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "peak": max(cpu_usage) if cpu_usage else 0,
                "minimum": min(cpu_usage) if cpu_usage else 0
            },
            "disk_usage": {
                "average": sum(disk_usage) / len(disk_usage) if disk_usage else 0,
                "peak": max(disk_usage) if disk_usage else 0,
                "minimum": min(disk_usage) if disk_usage else 0
            },
            "threshold_violations": {
                "total_count": len(all_violations),
                "unique_violations": list(set(all_violations))
            }
        }
        
        return summary
    
    def save_health_report(self, filename: Optional[str] = None) -> Path:
        """Save health monitoring report to file."""
        summary = self.get_health_summary()
        
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"system_health_report_{timestamp}.json"
        
        output_path = Path("logs") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "health_summary": summary,
            "detailed_data": self.health_data,
            "thresholds": self.thresholds
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"System health report saved to: {output_path}")
        return output_path
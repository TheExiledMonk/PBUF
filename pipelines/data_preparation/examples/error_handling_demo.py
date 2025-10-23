#!/usr/bin/env python3
"""
Demonstration of the comprehensive error handling and logging system.

This script shows how the enhanced error handling and transformation logging
systems work together to provide detailed error reports, audit trails, and
processing summaries.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.core.error_handling import (
    EnhancedProcessingError, ErrorRecoveryManager, ProcessingLogger,
    ProcessingStage, ErrorSeverity
)
from data_preparation.core.transformation_logging import (
    TransformationLogger, SystemHealthMonitor
)


def demonstrate_error_handling():
    """Demonstrate the enhanced error handling system."""
    print("=" * 60)
    print("ENHANCED ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    
    # Create a processing logger
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = ProcessingLogger(
            name="demo_logger",
            log_directory=Path(temp_dir) / "logs"
        )
        
        # Create an enhanced processing error
        error = EnhancedProcessingError(
            dataset_name="demo_dataset",
            stage=ProcessingStage.INPUT_VALIDATION,
            error_type="file_not_found",
            error_message="Input file could not be located",
            severity=ErrorSeverity.ERROR,
            context={
                "expected_path": "/path/to/missing/file.txt",
                "search_paths": ["/data", "/tmp", "/home/user"],
                "file_size_expected": 1024000
            },
            suggested_actions=[
                "Check if file path is correct",
                "Verify file permissions",
                "Re-download the dataset if necessary"
            ]
        )
        
        print("Enhanced Processing Error Created:")
        print("-" * 40)
        print(f"Dataset: {error.error_context.dataset_name}")
        print(f"Stage: {error.error_context.stage.value}")
        print(f"Severity: {error.error_context.severity.value}")
        print(f"Error Type: {error.error_context.error_type}")
        print()
        
        # Log the error
        logger.log_error(error)
        print("✓ Error logged with detailed context")
        
        # Demonstrate error recovery
        recovery_manager = ErrorRecoveryManager(logger.logger)
        
        # Register a custom recovery strategy
        def demo_recovery_strategy(error):
            print(f"  Attempting recovery for: {error.error_type}")
            # Simulate recovery attempt
            return False  # Recovery failed for demo
        
        recovery_manager.register_recovery_strategy("file_not_found", demo_recovery_strategy)
        
        success = recovery_manager.attempt_recovery(error)
        print(f"✓ Recovery attempted, success: {success}")
        
        # Show recovery statistics
        stats = recovery_manager.get_recovery_stats()
        print(f"✓ Recovery statistics: {stats}")
        
        print()
        print("Detailed Error Report:")
        print("-" * 40)
        print(error.generate_detailed_report())


def demonstrate_transformation_logging():
    """Demonstrate the transformation logging system."""
    print("\n" + "=" * 60)
    print("TRANSFORMATION LOGGING DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create transformation logger
        logger = TransformationLogger(
            dataset_name="demo_supernova_dataset",
            dataset_type="sn",
            output_directory=Path(temp_dir) / "logs"
        )
        
        # Start processing
        logger.start_processing({
            "source": "Pantheon+ Compilation",
            "version": "v1.0",
            "total_supernovae": 1701
        })
        
        # Demonstrate transformation steps
        with logger.log_transformation_step(
            step_name="magnitude_to_distance_modulus",
            formula=r"\mu = m - M",
            description="Convert apparent magnitude to distance modulus",
            input_description="Apparent magnitudes with systematic corrections",
            output_description="Distance moduli in standard format",
            references=["Betoule et al. 2014", "Scolnic et al. 2018"],
            assumptions=["Standard candle assumption", "Negligible peculiar velocities"]
        ) as step:
            # Log parameters used in transformation
            logger.log_parameter("absolute_magnitude_B", -19.3, "Absolute B-band magnitude")
            logger.log_parameter("systematic_correction", 0.02, "Applied systematic correction")
            
            # Simulate data shapes
            import numpy as np
            input_data = np.random.randn(1701, 3)  # z, m, σ_m
            output_data = np.random.randn(1701, 3)  # z, μ, σ_μ
            
            logger.log_data_shape(input_data, "input_magnitudes")
            logger.log_data_shape(output_data, "output_distance_moduli")
        
        # Another transformation step
        with logger.log_transformation_step(
            step_name="covariance_matrix_application",
            formula=r"C_{total} = C_{stat} + C_{sys}",
            description="Apply statistical and systematic covariance matrices",
            parameters={
                "statistical_only": False,
                "systematic_sources": ["calibration", "dust", "intrinsic"]
            }
        ) as step:
            logger.log_parameter("covariance_dimension", "1701x1701", "Full covariance matrix size")
            logger.log_parameter("condition_number", 1.2e6, "Matrix condition number")
        
        # Log validation results
        validation_results = {
            "schema_validation": {"passed": True, "details": {"all_fields_present": True}},
            "numerical_validation": {"passed": True, "details": {"no_nan_values": True, "finite_values": True}},
            "covariance_validation": {"passed": True, "details": {"positive_definite": True, "symmetric": True}}
        }
        logger.log_validation_results(validation_results)
        
        # End processing
        logger.end_processing({
            "output_format": "StandardDataset",
            "total_data_points": 1701,
            "processing_successful": True
        })
        
        print("✓ Transformation logging completed")
        
        # Generate summaries
        processing_summary = logger.generate_processing_summary()
        print(f"✓ Processing summary generated: {len(processing_summary.transformation_steps)} steps")
        
        publication_summary = logger.generate_publication_summary()
        print(f"✓ Publication summary generated")
        
        # Save summaries
        summary_path = logger.save_processing_summary()
        pub_path = logger.save_publication_summary()
        latex_path = logger.save_latex_summary()
        
        print(f"✓ Summaries saved:")
        print(f"  - Processing: {summary_path}")
        print(f"  - Publication: {pub_path}")
        print(f"  - LaTeX: {latex_path}")
        
        print()
        print("Publication Summary:")
        print("-" * 40)
        import json
        print(json.dumps(publication_summary, indent=2))


def demonstrate_system_health_monitoring():
    """Demonstrate the system health monitoring."""
    print("\n" + "=" * 60)
    print("SYSTEM HEALTH MONITORING DEMONSTRATION")
    print("=" * 60)
    
    monitor = SystemHealthMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    print("✓ System health monitoring started")
    
    # Record several snapshots
    import time
    for i in range(3):
        snapshot = monitor.record_health_snapshot()
        print(f"✓ Health snapshot {i+1}: {snapshot.get('timestamp', 'no timestamp')}")
        if 'memory' in snapshot:
            print(f"  Memory usage: {snapshot['memory']['percent']:.1f}%")
        if 'cpu' in snapshot:
            print(f"  CPU usage: {snapshot['cpu']['percent']:.1f}%")
        time.sleep(0.1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("✓ System health monitoring stopped")
    
    # Get health summary
    summary = monitor.get_health_summary()
    print("✓ Health summary generated:")
    
    if 'memory_usage' in summary:
        print(f"  Average memory usage: {summary['memory_usage']['average']:.1f}%")
        print(f"  Peak memory usage: {summary['memory_usage']['peak']:.1f}%")
    
    if 'threshold_violations' in summary:
        violations = summary['threshold_violations']['total_count']
        print(f"  Threshold violations: {violations}")
    
    # Save health report
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = SystemHealthMonitor()
        monitor.start_monitoring()
        monitor.record_health_snapshot()
        monitor.stop_monitoring()
        
        report_path = monitor.save_health_report()
        print(f"✓ Health report saved to: {report_path}")


def main():
    """Run all demonstrations."""
    print("COMPREHENSIVE ERROR HANDLING AND LOGGING DEMONSTRATION")
    print("This demonstrates the enhanced error handling and transformation logging")
    print("systems implemented for the data preparation framework.")
    print()
    
    try:
        demonstrate_error_handling()
        demonstrate_transformation_logging()
        demonstrate_system_health_monitoring()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The comprehensive error handling and logging system provides:")
        print("• Enhanced error reporting with detailed context")
        print("• Automatic error recovery mechanisms")
        print("• Detailed transformation logging with formula references")
        print("• Processing summaries suitable for publication")
        print("• System health monitoring and performance tracking")
        print("• Complete audit trails for reproducibility")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
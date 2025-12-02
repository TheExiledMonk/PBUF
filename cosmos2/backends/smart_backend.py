#!/usr/bin/env python3

import numpy as np
import time
import sys
sys.path.append('/home/fabian/PBUF5-GPU')

from cosmos2.backends.rocm_backend import ROCmBackend

class SmartGPUBackend:
    """Smart GPU backend with optimal performance thresholds based on benchmarks."""
    
    def __init__(self):
        self.backend = ROCmBackend()
        self.performance_stats = {
            'matrix_multiply': {'gpu_calls': 0, 'cpu_calls': 0, 'gpu_time': 0, 'cpu_time': 0},
            'matrix_inverse': {'gpu_calls': 0, 'cpu_calls': 0, 'gpu_time': 0, 'cpu_time': 0}
        }
        
        # Optimal thresholds based on benchmark results
        self.thresholds = {
            'matrix_multiply': 200,  # GPU beneficial for ≥200x200
            'matrix_inverse': 100    # GPU beneficial for ~100x100 only
        }
        
        print("[smart] Smart GPU backend initialized with optimal thresholds")
        print(f"[smart] Matrix multiply: GPU for ≥{self.thresholds['matrix_multiply']}x{self.thresholds['matrix_multiply']}")
        print(f"[smart] Matrix inverse: GPU for ~{self.thresholds['matrix_inverse']}x{self.thresholds['matrix_inverse']}")
        
        # Warm up GPU
        self._warmup_gpu()
    
    def _warmup_gpu(self):
        """Warm up GPU to eliminate initialization overhead."""
        print("[smart] Warming up GPU...")
        try:
            A = np.random.uniform(0.1, 10.0, (10, 10))
            B = np.random.uniform(0.1, 10.0, (10, 10))
            
            self.backend.matrix_multiply(A, B)
            self.backend.matrix_inverse(A)
            
            print("[smart] GPU warmed up successfully")
        except Exception as e:
            print(f"[smart] GPU warmup failed: {e}")
    
    def _make_well_conditioned(self, matrix):
        """Make a matrix well-conditioned for inversion."""
        n = matrix.shape[0]
        for i in range(n):
            matrix[i, i] += np.sum(np.abs(matrix[i, :])) + 1.0
        return matrix
    
    def matrix_multiply(self, A, B):
        """Smart matrix multiplication with optimal algorithm selection."""
        size = max(A.shape[0], A.shape[1])
        operation = 'matrix_multiply'
        
        # Decision: GPU for large matrices, CPU for small ones
        use_gpu = size >= self.thresholds[operation]
        
        if use_gpu:
            try:
                start_time = time.time()
                result = self.backend.matrix_multiply(A, B)
                gpu_time = time.time() - start_time
                
                # Verify accuracy
                start_time = time.time()
                cpu_result = np.matmul(A, B)
                cpu_time = time.time() - start_time
                
                accuracy = np.max(np.abs(result - cpu_result))
                
                if accuracy < 1e-6:  # Acceptable accuracy for multiply
                    self.performance_stats[operation]['gpu_calls'] += 1
                    self.performance_stats[operation]['gpu_time'] += gpu_time
                    return result
                else:
                    print(f"[smart] GPU multiply accuracy poor ({accuracy:.2e}), using CPU")
                    self.performance_stats[operation]['cpu_calls'] += 1
                    self.performance_stats[operation]['cpu_time'] += cpu_time
                    return cpu_result
                    
            except Exception as e:
                print(f"[smart] GPU multiply failed: {e}, using CPU")
                use_gpu = False
        
        # CPU path
        start_time = time.time()
        result = np.matmul(A, B)
        cpu_time = time.time() - start_time
        
        if not use_gpu:
            self.performance_stats[operation]['cpu_calls'] += 1
            self.performance_stats[operation]['cpu_time'] += cpu_time
        
        return result
    
    def matrix_inverse(self, A):
        """Smart matrix inverse with optimal algorithm selection."""
        size = A.shape[0]
        operation = 'matrix_inverse'
        
        # Decision: GPU only for ~100x100 matrices
        use_gpu = 50 <= size <= 150  # Sweet spot around 100x100
        
        if use_gpu:
            try:
                start_time = time.time()
                result = self.backend.matrix_inverse(A)
                gpu_time = time.time() - start_time
                
                # Verify accuracy
                identity_check = np.dot(A, result)
                accuracy = np.max(np.abs(identity_check - np.eye(size)))
                
                if accuracy < 1e-10:  # High accuracy for inverse
                    self.performance_stats[operation]['gpu_calls'] += 1
                    self.performance_stats[operation]['gpu_time'] += gpu_time
                    return result
                else:
                    print(f"[smart] GPU inverse accuracy poor ({accuracy:.2e}), using CPU")
                    self.performance_stats[operation]['cpu_calls'] += 1
                    self.performance_stats[operation]['cpu_time'] += gpu_time  # Count the failed GPU time
                    return np.linalg.inv(A)
                    
            except Exception as e:
                print(f"[smart] GPU inverse failed: {e}, using CPU")
                use_gpu = False
        
        # CPU path
        start_time = time.time()
        result = np.linalg.inv(A)
        cpu_time = time.time() - start_time
        
        if not use_gpu:
            self.performance_stats[operation]['cpu_calls'] += 1
            self.performance_stats[operation]['cpu_time'] += cpu_time
        
        return result
    
    def get_performance_summary(self):
        """Get performance summary."""
        summary = {}
        
        for operation, stats in self.performance_stats.items():
            total_calls = stats['gpu_calls'] + stats['cpu_calls']
            if total_calls > 0:
                gpu_usage = stats['gpu_calls'] / total_calls
                avg_gpu_time = stats['gpu_time'] / stats['gpu_calls'] if stats['gpu_calls'] > 0 else 0
                avg_cpu_time = stats['cpu_time'] / stats['cpu_calls'] if stats['cpu_calls'] > 0 else 0
                avg_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0
                
                summary[operation] = {
                    'total_calls': total_calls,
                    'gpu_usage': f"{gpu_usage:.1%}",
                    'avg_gpu_time': f"{avg_gpu_time:.6f}s",
                    'avg_cpu_time': f"{avg_cpu_time:.6f}s",
                    'avg_speedup': f"{avg_speedup:.2f}x"
                }
        
        return summary

def test_smart_backend():
    """Test the smart backend with various workloads."""
    print("=== Smart GPU Backend Test ===")
    
    backend = SmartGPUBackend()
    
    # Test workloads
    test_workloads = [
        (25, "Small workload"),
        (75, "Medium workload"),
        (150, "Large workload"),
        (250, "Very large workload"),
        (350, "Extra large workload")
    ]
    
    print("\n🧪 Performance Testing:")
    print("Size     Multiply    Inverse     Algorithm Choice")
    print("-" * 55)
    
    for size, description in test_workloads:
        # Generate test matrices
        A = np.random.uniform(0.1, 10.0, (size, size))
        A = backend._make_well_conditioned(A)
        B = np.random.uniform(0.1, 10.0, (size, size))
        B = backend._make_well_conditioned(B)
        
        # Test operations
        start = time.time()
        C = backend.matrix_multiply(A, B)
        mult_time = time.time() - start
        
        start = time.time()
        A_inv = backend.matrix_inverse(A)
        inv_time = time.time() - start
        
        # Determine algorithm choices
        mult_algo = "GPU" if size >= backend.thresholds['matrix_multiply'] else "CPU"
        inv_algo = "GPU" if 50 <= size <= 150 else "CPU"
        
        print(f"{size:4d}    {mult_time:8.6f}s   {inv_time:8.6f}s   Mult:{mult_algo} Inv:{inv_algo}")
    
    # Show performance summary
    print("\n📊 Performance Summary:")
    summary = backend.get_performance_summary()
    
    for operation, stats in summary.items():
        print(f"{operation}:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  GPU usage: {stats['gpu_usage']}")
        print(f"  Avg GPU time: {stats['avg_gpu_time']}")
        print(f"  Avg CPU time: {stats['avg_cpu_time']}")
        print(f"  Avg speedup: {stats['avg_speedup']}")

if __name__ == "__main__":
    test_smart_backend()

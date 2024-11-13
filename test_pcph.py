import torch
from torch import Tensor
from typing import Optional


import numpy as np
import time
from typing import Callable
import matplotlib.pyplot as plt
from functools import partial
import unittest

def generate_pcph_original(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 == 0.0):
        return noise

    vuv = f0 > 0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = sample_rate / 2.0 / f0[vuv]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    # Compute harmonic amplitude
    harmonic_amplitude = vuv * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to the harmonic signal
    harmonics = harmonic_mask * harmonics
    harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise

def generate_pcph_optimized(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude: Optional[float] = 0.01,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames).
        hop_length (int): Hop length of the F0 sequence.
        sample_rate (int): Sampling frequency of the waveform in Hz.
        noise_amplitude (float, optional): Amplitude of the noise component (default: 0.01).
        random_init_phase (bool, optional): Whether to initialize phases randomly (default: True).
        power_factor (float, optional): Factor to control the power of harmonics (default: 0.1).
        max_frequency (float, optional): Maximum frequency to define the number of harmonics (default: None).

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length).
    """
    batch, _, frames = f0.size()
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )
    if torch.all(f0 == 0.0):
        return noise

    vuv = f0 > 0
    min_f0_value = torch.min(f0[f0 > 0]).item()
    max_frequency = max_frequency if max_frequency is not None else sample_rate / 2
    max_n_harmonics = int(max_frequency / min_f0_value)
    n_harmonics = torch.ones_like(f0, dtype=torch.float)
    n_harmonics[vuv] = sample_rate / 2.0 / f0[vuv]

    indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
    harmonic_f0 = f0 * indices

    # Compute harmonic mask
    harmonic_mask = harmonic_f0 <= (sample_rate / 2.0)
    harmonic_mask = torch.repeat_interleave(harmonic_mask, hop_length, dim=2)

    # Compute harmonic amplitude
    harmonic_amplitude = vuv * power_factor * torch.sqrt(2.0 / n_harmonics)
    harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)

    # Generate sinusoids
    f0 = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = f0.to(torch.float64) / sample_rate
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious = torch.cumsum(radious, dim=2)
    harmonic_phase = 2.0 * torch.pi * radious * indices
    harmonics = torch.sin(harmonic_phase).to(torch.float32)

    # Multiply coefficients to the harmonic signal
    harmonics = harmonic_mask * harmonics
    harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

    return harmonics + noise

class TestPCPHGenerator(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 44100
        self.hop_length = 256
        
    def generate_test_f0(self, batch_size: int, frames: int) -> torch.Tensor:
        """Generate test F0 sequences."""
        f0 = torch.zeros((batch_size, 1, frames), device=self.device)
        # Add some realistic F0 values (e.g., 100-400 Hz)
        f0[:, 0, 10:20] = torch.linspace(100, 400, 10, device=self.device)
        return f0

    def test_output_shape(self):
        """Test if both implementations produce correct output shapes."""
        batch_size, frames = 2, 30
        f0 = self.generate_test_f0(batch_size, frames)
        
        for impl in [generate_pcph_original, generate_pcph_optimized]:
            with self.subTest(implementation=impl.__name__):
                output = impl(f0, self.hop_length, self.sample_rate)
                expected_shape = (batch_size, 1, frames * self.hop_length)
                self.assertEqual(output.shape, expected_shape)

    def test_zero_f0(self):
        """Test behavior with zero F0 input."""
        batch_size, frames = 2, 30
        f0 = torch.zeros((batch_size, 1, frames), device=self.device)
        
        for impl in [generate_pcph_original, generate_pcph_optimized]:
            with self.subTest(implementation=impl.__name__):
                output = impl(f0, self.hop_length, self.sample_rate)
                # Should return noise only
                self.assertTrue(torch.abs(output).mean() < 0.1)

    def test_output_consistency(self):
        """Test if both implementations produce similar outputs."""
        batch_size, frames = 2, 30
        f0 = self.generate_test_f0(batch_size, frames)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        output1 = generate_pcph_original(f0, self.hop_length, self.sample_rate)
        
        torch.manual_seed(42)
        output2 = generate_pcph_optimized(f0, self.hop_length, self.sample_rate)
        
        # Check if outputs are close (allowing for small numerical differences)
        self.assertTrue(torch.allclose(output1, output2, rtol=1e-4, atol=1e-4))

def benchmark_implementation(
    impl: Callable,
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    n_runs: int = 5
) -> float:
    """Benchmark a single implementation."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(n_runs):
        output = impl(f0, hop_length, sample_rate)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    end_time = time.perf_counter()
    return (end_time - start_time) / n_runs

def run_benchmarks(
    batch_sizes: list[int],
    frame_lengths: list[int],
    n_runs: int = 5
) -> dict:
    """Run comprehensive benchmarks for both implementations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {
        'original': {},
        'optimized': {}
    }
    
    for batch_size in batch_sizes:
        for frames in frame_lengths:
            print(f"\nBenchmarking batch_size={batch_size}, frames={frames}")
            
            # Generate test data
            f0 = torch.zeros((batch_size, 1, frames), device=device)
            f0[:, 0, 10:20] = torch.linspace(100, 400, 10, device=device)
            
            # Benchmark both implementations
            for name, impl in [
                ('original', generate_pcph_original),
                ('optimized', generate_pcph_optimized)
            ]:
                # Warmup run
                impl(f0, 256, 44100)
                
                # Actual benchmark
                time_taken = benchmark_implementation(
                    impl, f0, 256, 44100, n_runs=n_runs
                )
                results[name][f"{batch_size}_{frames}"] = time_taken
                print(f"{name}: {time_taken:.4f} seconds")
    
    return results

def plot_benchmark_results(results: dict, batch_sizes: list[int], frame_lengths: list[int]):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(batch_sizes) * len(frame_lengths))
    width = 0.35
    
    original_times = []
    optimized_times = []
    labels = []
    
    for batch_size in batch_sizes:
        for frames in frame_lengths:
            key = f"{batch_size}_{frames}"
            original_times.append(results['original'][key])
            optimized_times.append(results['optimized'][key])
            labels.append(f"B{batch_size}\nF{frames}")
    
    plt.bar(x - width/2, original_times, width, label='Original')
    plt.bar(x + width/2, optimized_times, width, label='Optimized')
    
    plt.xlabel('Batch Size (B) and Frame Length (F)')
    plt.ylabel('Time (seconds)')
    plt.title('PCPH Generator Performance Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('pcph_benchmark_results.png')
    plt.close()

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False)
    
    # Run benchmarks with various batch sizes and frame lengths
    print("\nRunning benchmarks...")
    batch_sizes = [1, 2, 4, 8]
    frame_lengths = [25, 50, 100, 200]
    
    results = run_benchmarks(batch_sizes, frame_lengths)
    
    # Plot results
    plot_benchmark_results(results, batch_sizes, frame_lengths)
    
    # Print speedup statistics
    print("\nSpeedup Statistics:")
    speedups = []
    for key in results['original'].keys():
        speedup = results['original'][key] / results['optimized'][key]
        speedups.append(speedup)
        print(f"Configuration {key}: {speedup:.2f}x speedup")
    
    print(f"\nAverage speedup: {np.mean(speedups):.2f}x")
    print(f"Maximum speedup: {np.max(speedups):.2f}x")
    print(f"Minimum speedup: {np.min(speedups):.2f}x")

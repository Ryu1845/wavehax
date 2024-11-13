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
    batch, _, frames = f0.size()  # f0: (batch, 1, frames)
    device = f0.device
    noise = noise_amplitude * torch.randn(
        (batch, 1, frames * hop_length), device=device
    )  # noise: (batch, 1, frames * hop_length)
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

    # Pre-allocate expanded f0 and compute phases
    f0_expanded = torch.repeat_interleave(f0, hop_length, dim=2)
    radious = (f0_expanded.to(torch.float64) / sample_rate).contiguous()
    if random_init_phase:
        radious[..., 0] += torch.rand((1, 1), device=device)
    radious.cumsum_(dim=2)  # In-place cumsum
    harmonic_phase = 2.0 * np.pi * radious * indices
    
    # Generate sinusoids with combined mask and amplitude
    harmonics = torch.sin(harmonic_phase).to(torch.float32)
    
    # Combine mask and amplitude calculations
    mask_amp = (harmonic_f0 <= (sample_rate / 2.0)).to(torch.float32)
    mask_amp *= (vuv * power_factor * torch.sqrt(2.0 / n_harmonics))
    mask_amp = torch.repeat_interleave(mask_amp, hop_length, dim=2)
    
    # Apply mask and sum
    harmonics.mul_(mask_amp)
    harmonics = harmonics.sum(dim=1, keepdim=True)

    return harmonics + noise

@torch.jit.script
def generate_pcph_optimized(
    f0: Tensor,  # Shape: (batch, 1, frames)
    hop_length: int,
    sample_rate: int,
    noise_amplitude: float = 0.01,
    random_init_phase: bool = True,
    power_factor: float = 0.1,
) -> Tensor:  # Shape: (batch, 1, frames * hop_length)
    """
    Generate pseudo-constant-power harmonic waveforms based on input F0 sequences.
    The spectral envelope of harmonics is designed to have flat spectral envelopes.

    Shape:
        - f0: (batch, 1, frames)
        - Output: (batch, 1, frames * hop_length)
        
        Internal shapes:
        - noise: (batch, 1, frames * hop_length)
        - vuv: (batch, 1, frames)
        - n_harmonics: (batch, 1, frames)
        - indices: (1, max_n_harmonics, 1)
        - harmonic_f0: (batch, max_n_harmonics, frames)
        - harmonic_mask: (batch, max_n_harmonics, frames)
        - harmonic_amplitude: (batch, 1, frames)
        - phase_cumsum: (batch, 1, frames * hop_length)
        - harmonics: (batch, max_n_harmonics, frames * hop_length)

    Args:
        f0 (Tensor): F0 sequences with shape (batch, 1, frames)
        hop_length (int): Hop length of the F0 sequence
        sample_rate (int): Sampling frequency of the waveform in Hz
        noise_amplitude (float, optional): Amplitude of the noise component. Default: 0.01
        random_init_phase (bool, optional): Whether to initialize phases randomly. Default: True
        power_factor (float, optional): Factor to control the power of harmonics. Default: 0.1

    Returns:
        Tensor: Generated harmonic waveform with shape (batch, 1, frames * hop_length)
    """
    batch, _, frames = f0.size()
    device = f0.device
    nyquist = sample_rate * 0.5

    noise = noise_amplitude * torch.randn((batch, 1, frames * hop_length), device=device)
    
    # Early return for all-zero F0
    if torch.all(f0 == 0.0):
        return noise

    # Voice/unvoiced flag and harmonic calculations
    vuv = f0 > 0.0  # (batch, 1, frames)
    min_f0 = torch.min(f0[vuv]).item()
    max_harmonics = int(nyquist / min_f0)
    
    # Pre-compute indices once - (1, max_harmonics, 1)
    indices = torch.arange(1, max_harmonics + 1, device=device).reshape(1, -1, 1)
    
    # Compute harmonics and mask in one step - (batch, max_harmonics, frames)
    harmonic_f0 = f0 * indices
    harmonic_mask = (harmonic_f0 <= nyquist) & vuv.expand(-1, max_harmonics, -1)
    
    # Compute amplitude factors - (batch, 1, frames)
    n_harmonics = torch.where(vuv, nyquist / f0, torch.ones_like(f0))
    harmonic_amplitude = (power_factor * torch.sqrt(2.0 / n_harmonics)) * vuv
    
    # Phase computation with improved numerical stability
    f0_expanded = torch.repeat_interleave(f0, hop_length, dim=2)  # (batch, 1, frames * hop_length)
    phase_cumsum = (f0_expanded / sample_rate).cumsum(dim=2)  # (batch, 1, frames * hop_length)
    
    if random_init_phase:
        phase_cumsum = phase_cumsum + torch.rand((batch, 1, 1), device=device)
    
    # Generate all harmonics at once with broadcasting
    phase_harmonics = 2.0 * torch.pi * phase_cumsum * indices  # (batch, max_harmonics, frames * hop_length)
    harmonics = torch.sin(phase_harmonics.to(torch.float32))
    
    # Apply mask and amplitude in frequency domain
    harmonic_mask = torch.repeat_interleave(harmonic_mask.to(harmonics.dtype), hop_length, dim=2)
    harmonics = harmonics * harmonic_mask
    
    # Sum harmonics and apply amplitude
    harmonic_amplitude = torch.repeat_interleave(harmonic_amplitude, hop_length, dim=2)
    harmonics = harmonic_amplitude * harmonics.sum(dim=1, keepdim=True)
    
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
        
        for impl, name in ((generate_pcph_original, "original"), (generate_pcph_optimized, "optimized")):
            with self.subTest(implementation=name):
                output = impl(f0, self.hop_length, self.sample_rate)
                expected_shape = (batch_size, 1, frames * self.hop_length)
                self.assertEqual(output.shape, expected_shape)

    def test_zero_f0(self):
        """Test behavior with zero F0 input."""
        batch_size, frames = 2, 30
        f0 = torch.zeros((batch_size, 1, frames), device=self.device)
        
        for impl, name in ((generate_pcph_original, "original"), (generate_pcph_optimized, "optimized")):
            with self.subTest(implementation=name):
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
        
        self.assertTrue((output1-output2).abs().mean()<0.01)

def benchmark_implementation(
    impl: Callable,
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    n_runs: int = 5,
) -> float:
    """Benchmark a single implementation.
    
    Args:
        impl: The implementation function to benchmark
        f0: Input F0 tensor
        hop_length: Hop length parameter
        sample_rate: Sample rate parameter
        n_runs: Number of timed runs to average over
    """
    # Actual benchmark runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for _ in range(n_runs):
        _ = impl(f0, hop_length, sample_rate)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    end_time = time.perf_counter()
    return (end_time - start_time) / n_runs

def run_benchmarks(
    batch_sizes: list[int],
    frame_lengths: list[int],
    n_runs: int = 5,
    n_iterations: int = 10
) -> dict:
    """Run comprehensive benchmarks for both implementations with statistical analysis.
    
    Args:
        batch_sizes: List of batch sizes to test
        frame_lengths: List of frame lengths to test
        n_runs: Number of runs per timing measurement
        n_iterations: Number of iterations to gather statistics
    
    Returns:
        Dictionary containing timing statistics for each configuration
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {
        'original': {'times': {}, 'stats': {}},
        'optimized': {'times': {}, 'stats': {}}
    }
    
    # Do warmup runs once per batch size
    for batch_size in batch_sizes:
        print(f"\nWarming up for batch_size={batch_size}")
        # Use the smallest frame length for warmup
        frames = frame_lengths[0]
        f0_warmup = torch.zeros((batch_size, 1, frames), device=device)
        f0_warmup[:, 0, 10:20] = torch.linspace(100, 400, 10, device=device)
        
        for impl in [generate_pcph_original, generate_pcph_optimized]:
            # Warmup runs
            for _ in range(3):  # 3 warmup runs
                _ = impl(f0_warmup, 256, 44100)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
    
        # Actual benchmarks for this batch size
        for frames in frame_lengths:
            key = f"{batch_size}_{frames}"
            print(f"\nBenchmarking batch_size={batch_size}, frames={frames}")
            
            # Benchmark both implementations
            for name, impl in [
                ('original', generate_pcph_original),
                ('optimized', generate_pcph_optimized)
            ]:
                torch.manual_seed(42)
                # Collect multiple timing measurements
                times = []
                for i in range(n_iterations):
                    time_taken = 0.0
                    for _ in range(n_runs):
                        # Generate test data
                        actual_frames = frames + torch.randint(-5, 6, (1,)).item()  # ±5 frames variation
                        f0 = torch.randn((batch_size, 1, actual_frames), device=device)*200+300
                        f0[f0<100]=0.0

                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = time.perf_counter()

                        _ = impl(f0, 256, 44100)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None

                        end_time = time.perf_counter()
                        time_taken += (end_time - start_time) / n_runs
                    times.append(time_taken)
                    print(f"{name} frames:{actual_frames} iteration {i+1}/{n_iterations}: {time_taken:.4f} seconds")
                
                times = np.array(times)
                
                # Calculate statistics
                mean_time = np.mean(times)
                std_time = np.std(times)
                ci_95 = 1.96 * std_time / np.sqrt(n_iterations)  # 95% confidence interval
                
                results[name]['times'][key] = times
                results[name]['stats'][key] = {
                    'mean': mean_time,
                    'std': std_time,
                    'ci_95': ci_95,
                    'actual_frames': frames
                }
                
                print(f"{name} statistics:")
                print(f"  Mean: {mean_time:.4f} ± {ci_95:.4f} seconds")
                print(f"  Std:  {std_time:.4f} seconds")
    
    return results

def plot_benchmark_results(results: dict, batch_sizes: list[int], frame_lengths: list[int]):
    """Plot benchmark results with error bars."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(batch_sizes) * len(frame_lengths))
    width = 0.35
    
    original_times = []
    original_errors = []
    optimized_times = []
    optimized_errors = []
    labels = []
    
    for batch_size in batch_sizes:
        for frames in frame_lengths:
            key = f"{batch_size}_{frames}"
            
            # Get mean times and confidence intervals
            original_times.append(results['original']['stats'][key]['mean'])
            original_errors.append(results['original']['stats'][key]['ci_95'])
            
            optimized_times.append(results['optimized']['stats'][key]['mean'])
            optimized_errors.append(results['optimized']['stats'][key]['ci_95'])
            
            actual_frames = results['original']['stats'][key].get('actual_frames', frames)
            labels.append(f"B{batch_size}\nF{actual_frames}")
    
    plt.bar(x - width/2, original_times, width, label='Original',
            yerr=original_errors, capsize=5)
    plt.bar(x + width/2, optimized_times, width, label='Optimized',
            yerr=optimized_errors, capsize=5)
    
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
    speedup_cis = []  # Confidence intervals for speedups
    
    for key in results['original']['stats'].keys():
        orig_stats = results['original']['stats'][key]
        opt_stats = results['optimized']['stats'][key]
        
        # Calculate speedup and its uncertainty
        speedup = orig_stats['mean'] / opt_stats['mean']
        
        # Error propagation for division
        rel_error = np.sqrt(
            (orig_stats['std']/orig_stats['mean'])**2 + 
            (opt_stats['std']/opt_stats['mean'])**2
        )
        speedup_ci = speedup * rel_error * 1.96  # 95% CI
        
        speedups.append(speedup)
        speedup_cis.append(speedup_ci)
        
        print(f"Configuration {key}: {speedup:.2f}x ± {speedup_ci:.2f}x speedup")
    
    mean_speedup = np.mean(speedups)
    std_speedup = np.std(speedups)
    ci_speedup = 1.96 * std_speedup / np.sqrt(len(speedups))
    
    print(f"\nOverall speedup: {mean_speedup:.2f}x ± {ci_speedup:.2f}x")
    print(f"Maximum speedup: {np.max(speedups):.2f}x")
    print(f"Minimum speedup: {np.min(speedups):.2f}x")

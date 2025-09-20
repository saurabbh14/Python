import matplotlib.pyplot as plt
import numpy as np

# Spatial grid
class RGrid:
    def __init__(self, rmin: float, rmax: float, ngrid: int):
        self.rmin = rmin
        self.rmax = rmax
        self.nr = ngrid
        self.dR = (rmax - rmin) / ngrid
        self.len = rmax - rmin
        self.dK = 2 * np.pi / self.len
        self.kmin = -np.pi / self.dR
        self.kmax = np.pi / self.dR
        self.cutOffe = np.square(self.kmax) / 2
        self.r_grid = None
        self.k_grid = None

    @classmethod
    def from_length(cls, length: float, ngrid: int):
        grd_min = -length / 2
        grd_max = length / 2
        return cls(grd_min, grd_max, ngrid)

    @classmethod
    def from_center_width(cls, center: float, width: float, ngrid: int):
        grd_min = center - width / 2
        grd_max = center + width / 2
        return cls(grd_min, grd_max, ngrid)

    def generate_grid(self) -> np.ndarray:
        self.r_grid = np.linspace(self.rmin, self.rmax - self.dR, self.nr)
        return self.r_grid

    def get_rgrid(self) -> np.ndarray:
        if self.r_grid is None:
            self.generate_grid()
        return self.r_grid

    def get_kgrid(self) -> np.ndarray:
        """Get corresponding momentum/k-space grid for Fourier transforms."""
        if self.k_grid is None:
            # For FFT, use fftfreq for proper ordering
            k_values = np.fft.fftfreq(self.nr, d=self.dR) * 2 * np.pi
            self.k_grid = k_values
        return self.k_grid

    def get_shifted_momentum_grid(self) -> np.ndarray:
        return np.fft.fftshift(self.get_kgrid())

    def display_grid(self):
        if self.r_grid is None:
            print("Grid not generated yet.")
        else:
            print(f"Spatial Grid: {self.nr} points from {self.rmin} to {self.rmax}")
            print(f"Grid spacing: {self.dR:.6f}")
            print(f"First few points: {self.r_grid[:5]}")
            print(f"Last few points: {self.r_grid[-5:]}")

    def display_momentum_info(self):
        k_grid = self.get_kgrid()
        print(f"Momentum Grid: {len(k_grid)} points")
        print(f"dk: {self.dK:.6f}")
        print(f"kMax: {self.kmax:.6f}")
        print(f"k range: [{k_grid.min():.6f}, {k_grid.max():.6f}]")

    def save_grid_to_file(self, filename: str):
        if self.r_grid is None:
            self.generate_grid()
        np.savetxt(filename, self.r_grid, header=f"Spatial grid: {self.nr} points, dx={self.dR}")
        print(f"Grid saved to {filename}")

    def plot_grid(self, show_points: bool = True):
        grid = self.get_rgrid()
        plt.figure(figsize=(10, 4))
        if show_points:
            plt.plot(grid, np.zeros_like(grid), 'bo', markersize=3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Position')
        plt.title(f'Spatial Grid ({self.nr} points)')
        plt.grid(True, alpha=0.3)
        plt.show()


class TGrid:
    def __init__(self, tmin: float, tmax: float, nt: int):
        self.Tmin = tmin
        self.Tmax = tmax
        self.nt = nt
        self.dt = (tmax - tmin) / (nt - 1)
        self.duration = tmax - tmin
        self.grid = None
        self._omega_grid = None
        self._sample_rate = 1.0 / self.dt

    @classmethod
    def from_duration(cls, duration: float, nt: int):
        return cls(0, duration, nt)

    @classmethod
    def from_sample_rate(cls, sample_rate: float, duration: float):
        nt = int(sample_rate * duration) + 1
        return cls(0, duration, nt)

    def generate_grid(self) -> np.ndarray:
        self.grid = np.linspace(self.Tmin, self.Tmax, self.nt)
        return self.grid

    def get_grid(self) -> np.ndarray:
        if self.grid is None:
            self.generate_grid()
        return self.grid

    def get_frequency_grid(self) -> np.ndarray:
        if self._omega_grid is None:
            freq_values = np.fft.fftfreq(self.nt, d=self.dt) * 2 * np.pi
            self._omega_grid = freq_values
        return self._omega_grid

    def get_shifted_frequency_grid(self) -> np.ndarray:
        return np.fft.fftshift(self.get_frequency_grid())

    @property
    def nyquist_frequency(self) -> float:
        return np.pi / self.dt

    @property
    def frequency_resolution(self) -> float:
        return 2 * np.pi / self.duration

    def display_grid(self):
        if self.grid is None:
            print("Grid not generated yet.")
        else:
            print(f"Time Grid: {self.nt} points from {self.Tmin} to {self.Tmax}")
            print(f"Time step: {self.dt:.6f}")
            print(f"Sample rate: {self._sample_rate:.2f} Hz")
            print(f"First few points: {self.grid[:5]}")
            print(f"Last few points: {self.grid[-5:]}")

    def display_frequency_info(self):
        omega_grid = self.get_frequency_grid()
        print(f"Frequency Grid: {len(omega_grid)} points")
        print(f"dω: {self.frequency_resolution:.6f}")
        print(f"ω_max (Nyquist): {self.nyquist_frequency:.6f}")
        print(f"ω range: [{omega_grid.min():.6f}, {omega_grid.max():.6f}]")

    def save_grid_to_file(self, filename: str):
        if self.grid is None:
            self.generate_grid()
        np.savetxt(filename, self.grid, header=f"Time grid: {self.nt} points, dt={self.dt}")
        print(f"Grid saved to {filename}")

    def plot_grid(self, show_points: bool = True):
        grid = self.get_grid()
        plt.figure(figsize=(10, 4))
        if show_points:
            plt.plot(grid, np.zeros_like(grid), 'ro', markersize=3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Time')
        plt.title(f'Time Grid ({self.nt} points)')
        plt.grid(True, alpha=0.3)
        plt.show()


# Test functions for grid.py
def test_rgrid():
    print("=== Testing RGrid ===")
    r_grid = RGrid.from_length(20.0, 64)
    r_grid.display_grid()
    r_grid.display_momentum_info()
    r_grid.plot_grid()
    r_grid.save_grid_to_file("rgrid_test.txt")

    # Additional test: from_center_width
    r_grid2 = RGrid.from_center_width(0.0, 10.0, 64)
    assert r_grid2.rmin == -5.0
    assert r_grid2.rmax == 5.0
    assert r_grid2.nr == 64
    assert np.isclose(r_grid2.dR, 10./64.)
    print("RGrid from_center_width test passed.")

    # Test momentum grid
    k_grid = r_grid.get_kgrid()
    assert len(k_grid) == 64
    print("RGrid momentum grid test passed.")

def test_tgrid():
    print("\n=== Testing TGrid ===")
    t_grid = TGrid.from_duration(50.0, 101)
    t_grid.display_grid()
    t_grid.display_frequency_info()
    t_grid.plot_grid()
    t_grid.save_grid_to_file("tgrid_test.txt")

    # Additional test: from_sample_rate
    t_grid2 = TGrid.from_sample_rate(2.0, 50.0)
    print(t_grid2.dt)
    assert t_grid2.nt == 101
    assert np.isclose(t_grid2.dt, 0.5)
    print("TGrid from_sample_rate test passed.")

    # Test frequency grid
    omega_grid = t_grid.get_frequency_grid()
    assert len(omega_grid) == 101
    print("TGrid frequency grid test passed.")

def test_fourier_example():
    print("\n=== Fourier Transform Example ===")
    r_grid = RGrid.from_length(20.0, 256)
    x = r_grid.get_rgrid()
    k = r_grid.get_kgrid()

    # Gaussian function
    sigma = 2.0
    f_x = np.exp(-x ** 2 / (2 * sigma ** 2))

    # Fourier transform (normalized properly)
    f_k = np.fft.fft(f_x) * r_grid.dR / np.sqrt(2 * np.pi)

    # Analytical FT of Gaussian is Gaussian
    analytical_fk = sigma * np.exp(- (sigma * k)**2 / 2)

    # Check if close (up to shift and approximation)
    f_k_shifted = np.fft.fftshift(f_k)
    analytical_fk_shifted = np.fft.fftshift(analytical_fk)
    assert np.allclose(np.abs(f_k_shifted), analytical_fk_shifted, atol=1e-2)
    print("Fourier transform test passed (Gaussian FT matches analytical).")

# Run all tests
def test_grid_example():
    test_rgrid()
    test_fourier_example()
    test_tgrid()
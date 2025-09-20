from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from .grid import RGrid


class Potential(RGrid):
    def __init__(self, grid_min: float, grid_max: float, ngrid: int):
        RGrid.__init__(self, grid_min, grid_max, ngrid)
        self.r = self.get_rgrid()

        # Default parameters for each potential type
        self.parameters: Dict[str, Dict[str, Any]] = {
            'StepFunc': {'energy': 20.},
            'Harmonic': {'k': 1.0, 'cen': 0.0},
            'SoftCore': {'Za': 1.0, 'cen': 0.0, 'softP': 0.5},
            'SingleGaussian': {'expnt': 0.1424, 'strength': -0.64, 'cen': 0.0},
            'DoubleGaussian': {'expnt': 0.1424, 'strength': -0.64, 'cen': 1.0},
            'Rectangular': {'width': 2.0, 'height': 10.0},
            'MultiRectangular': {'width': 1.5, 'height': 5.0, 'num':2, 'distance': 3.0},
            'SuperGaussian': {'order':4, 'width': 1.5, 'strength': 5.0, 'num': 2, 'distance': 3.0},
            'Exponential': {'amplitude': 5.0, 'decay': 10.0},
            'Triangular': {'V0': 8.0, 'a': 1.0},
            'Polynomial': {'coef_vec': [0.0, 0.0, -1., 0., 1]}, # Double-well parameters
        }

        # Potential functions as static methods
        self._potential_functions = {
            'StepFunc': self._step_func,
            'Harmonic': self._harmonic_osc,
            'SoftCore': self._softcore_poten,
            'Polynomial': self._polynomial,
            'SingleGaussian': self._single_gaussian,
            'DoubleGaussian': self._double_gaussian,
            'Rectangular': self._rectangular,
            'Exponential': self._exponential,
            'Triangular': self._triangular
        }

        # Current potential type
        self.current_potential = 'Harmonic'

    @staticmethod
    def _softcore_poten(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        za = params['Za']
        cen = params['cen']
        soft_params = params['softP']
        return - za / np.sqrt((x - cen) ** 2 + soft_params ** 2)

    def _step_func(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        avgE = params['energy']
        return avgE * np.heaviside(x - self.len / 2, 1)

    @staticmethod
    def _harmonic_osc(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        k = params['k']
        cen = params['cen']
        return 0.5 * k * (x - cen)**2

    @staticmethod
    def _single_gaussian(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        expnt = params['expnt']
        v0 = params['strength']
        cen = params['cen']
        return v0 * np.sqrt(100 / np.pi) * np.exp(-expnt * (x - cen) ** 2)

    @staticmethod
    def _double_gaussian(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        cen = params['cen']
        expnt = params['expnt']
        v0 = params['strength']
        return (v0 * np.sqrt(10 / np.pi) * np.exp(- expnt * (x - cen) ** 2) +
                v0 * np.sqrt(10 / np.pi) * np.exp(- expnt * (x + cen) ** 2))

    @staticmethod
    def _rectangular(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        width = params['width']
        height = params['height']
        half_width = width / 2
        return height * ((x >= -half_width) & (x <= half_width)).astype(float)

    @staticmethod
    def _exponential(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        amplitude = params['amplitude']
        decay = params['decay']
        return amplitude * np.exp(-decay * np.abs(x))

    @staticmethod
    def _triangular(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        V0 = params['V0']
        a = params['a']
        abs_x = np.abs(x)
        return V0 * (1 - abs_x / a) * (abs_x <= a)

    @staticmethod
    def _polynomial(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        coef_vec = params['coef_vec']
        res = 0.0
        for coef in coef_vec:
            res = res * x + coef
        return res

    @classmethod
    def from_length(cls, length: float, ngrid: int):
        grid_res = RGrid.from_length(length, ngrid)
        return cls(grid_res.rmin, grid_res.rmax, ngrid)

    def set_potential(self, potential_type: str) -> None:
        if potential_type not in self._potential_functions:
            raise ValueError(f"Unknown potential type: {potential_type}. "
                             f"Available types: {list(self._potential_functions.keys())}")
        self.current_potential = potential_type

    def set_parameters(self, potential_type: Optional[str] = None, **kwargs) -> None:
        if potential_type is None:
            potential_type = self.current_potential

        if potential_type not in self.parameters:
            raise ValueError(f"Unknown potential type: {potential_type}")

        self.parameters[potential_type].update(kwargs)

    def get_parameters(self, potential_type: Optional[str] = None) -> Dict[str, Any]:
        if potential_type is None:
            potential_type = self.current_potential
        return self.parameters[potential_type].copy()

    def evaluate(self, x: Optional[np.ndarray] = None, potential_type: Optional[str] = None) -> np.ndarray:
        if x is None:
            x = self.r
        if potential_type is None:
            potential_type = self.current_potential

        func = self._potential_functions[potential_type]
        params = self.parameters[potential_type]
        return func(x, params)

    def value(self, x):
        func = self._potential_functions[self.current_potential]
        params = self.parameters[self.current_potential]
        return func(x, params)

    def plot(self, potential_types: Optional[List[str]] = None, figsize: Tuple[float, float] = (10, 6),
             title: Optional[str] = None, xlabel: str = 'Position (x)', ylabel: str = 'Potential V(x)',
             grid: bool = True, legend: bool = True) -> plt.Figure:

        if potential_types is None:
            potential_types = [self.current_potential]

        fig, ax = plt.subplots(figsize=figsize)

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for i, pot_type in enumerate(potential_types):
            V = self.evaluate(potential_type=pot_type)
            color = colors[i % len(colors)]
            params = self.parameters[pot_type]
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            label = f"{pot_type.replace('_', ' ').title()} ({param_str})"

            ax.plot(self.r, V, color=color, linewidth=2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Quantum Potential Wells')

        if grid:
            ax.grid(True, alpha=0.3)
        if legend and len(potential_types) > 1:
            ax.legend()

        plt.tight_layout()
        return fig

    def compare_all(self, figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        return self.plot(potential_types=list(self._potential_functions.keys()),
                         figsize=figsize, title='Comparison of All Potential Wells')

    @staticmethod
    def get_potential_info() -> Dict[str, str]:
        info = {
            'Harmonic': 'Harmonic oscillator potential',
            'SoftCore': 'Soft-core Coulomb potential',
            'SingleGaussian': 'Single Gaussian well/barrier',
            'DoubleGaussian': 'Double Gaussian wells/barriers separated by distance 2*cen',
            'Rectangular': 'Rectangular (square) well/barrier',
            'Exponential': 'Exponential decay well/barrier',
            'Triangular': 'Triangular well/barrier',
            'Polynomial': 'Polynomial potential defined by coefficients'
        }
        return info

    def __str__(self) -> str:
        current_params = self.get_parameters()
        param_str = ', '.join([f"{k}={v}" for k, v in current_params.items()])
        return (f"Potential(current='{self.current_potential}', "
                f"params={{{param_str}}}, x_range={self.len}, "
                f"points={self.nr})")

    def __repr__(self) -> str:
        return self.__str__()


class WavePacket(RGrid):
    def __init__(self, rmin: float, rmax: float, ngrid: int):
        super().__init__(rmin, rmax, ngrid)
        self.x0 = None
        self.p0 = None
        self.sigma0 = None

    def set_initial(self, x0: float, p0: float, sigma0: float):
        self.x0 = x0
        self.p0 = p0
        self.sigma0 = sigma0

    def value(self, x, phase):
        sigma2 = self.sigma0**2
        norm = (2 * np.pi * sigma2)**(-0.25)
        return norm * np.exp(-(x - self.x0)**2 / (4 * sigma2) + 1j * self.p0 * (x- self.x0) + 1j * phase)

    def get_wp(self, phi) -> np.ndarray:
        r = self.get_rgrid()
        psi = self.value(r, phi)
        # --- Discrete renormalization ---
        dx = r[1] - r[0]  # grid spacing
        norm_factor = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        psi /= norm_factor
        return psi

    def plot_wavefunction(self, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        psi = self.get_wp(0.1)
        ax.plot(self.get_rgrid(), np.abs(psi) ** 2, label='|ψ|^2')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Initial Wave Packet')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig


# Test functions for gridFunc.py
def test_potential_functions():
    print("=== Testing Potential ===")
    wells = Potential(-5.0, 5.0, 100)
    print("Current potential:", wells.current_potential)
    print("Current parameters:", wells.get_parameters())

    # Test polynomial
    wells.set_potential('Polynomial')
    wells.set_parameters(coef_vec=[0.5, 0.0, 1.0])  # 0.5 x^2 + 1.0
    V_poly = wells.evaluate()
    assert np.isclose(V_poly[50], 1.0)  # At x=0
    print("Polynomial evaluation test passed.")

    # Test set_potential and set_parameters
    wells.set_potential('DoubleGaussian')
    wells.set_parameters(expnt=0.2, strength=-1.0, cen=1.5)
    V_double = wells.evaluate()
    assert V_double.shape == (100,)
    print("DoubleGaussian evaluation test passed.")

    wells.set_potential('Rectangular')
    wells.set_parameters(width=2.5, height=8.0)
    V_rect = wells.evaluate()
    assert np.max(V_rect) == 8.0
    print("Rectangular evaluation test passed.")

    # Test triangular correction
    wells.set_potential('Triangular')
    V_tri = wells.evaluate()
    x = wells.r
    idx_center = np.argmin(np.abs(x))
    assert np.isclose(V_tri[idx_center], 8.0)  # V0 at center
    print("Triangular evaluation test passed (height corrected).")

    # Test plot and compare_all
    wells.plot(title="Triangular Potential")
    wells.compare_all()
    print("Plot tests passed (figures generated).")

    # Test evaluate on custom x
    x_test = np.array([-1, 0, 1])
    V_values = wells.evaluate(x_test)
    assert len(V_values) == 3
    print(f"Potential values at x={x_test}: V={V_values}")

    # Test info
    info = wells.get_potential_info()
    assert 'Harmonic' in info
    print("get_potential_info test passed.")

    # Test str
    print(wells)
    print("str/repr test passed.")

def test_wave_packet():
    print("\n=== Testing WavePacket ===")
    wp = WavePacket.from_length(100, 200)
    wp.set_initial(10.0, 1.0, 2.0)
    psi = wp.get_wp(np.pi/2)
    assert psi.shape == (200,)
    prob = np.abs(psi)**2
    integral = np.trapezoid(prob, wp.get_rgrid())
    assert np.isclose(integral, 1.0, atol=1e-3)  # Normalized
    print("Wavefunction normalization test passed.")

    wp.plot_wavefunction()
    print("WavePacket plot test passed.")

# Run all tests
def test_potential_example():
    test_potential_functions()
    test_wave_packet()
    plt.show()  # Show all plots at the end
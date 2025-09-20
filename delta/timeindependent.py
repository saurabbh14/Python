import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from potential.grid import RGrid
from potential.gridFunc import Potential


class Kinetic(RGrid):
    def __init__(self, rmin: float, rmax: float, ngrid: int):
        super().__init__(rmin, rmax, ngrid)

    def get_dvr_kinetic(self) -> sp.spmatrix:
        h = (self.rmax - self.rmin) / (self.nr - 1)

        # Index difference matrix
        i = np.arange(self.nr)
        j = np.arange(self.nr)
        diff = i[:, None] - j[None, :]

        # Initialize kinetic energy matrix
        T = np.zeros((self.nr, self.nr), dtype=float)

        # Diagonal elements: π²/(6h²)
        np.fill_diagonal(T, (np.pi ** 2) / (6 * h ** 2))

        # Off-diagonal elements: (-1)^(i-j) / (h²(i-j)²)
        mask = diff != 0
        T[mask] = ((-1.0) ** (diff[mask])) / (h ** 2 * diff[mask] ** 2)

        # Convert to sparse matrix for consistency
        return sp.csr_matrix(T)

    def get_finite_diff_kinetic(self, order: int = 2) -> sp.spmatrix:
        dx2 = self.dR ** 2

        if order == 2:
            main = -2 * np.ones(self.nr)
            off = np.ones(self.nr - 1)
            laplacian = sp.diags([off, main, off], [-1, 0, 1]) / dx2
        elif order == 4:
            main = -30 * np.ones(self.nr)
            off1 = 16 * np.ones(self.nr - 1)
            off2 = -1 * np.ones(self.nr - 2)
            laplacian = sp.diags([off2, off1, main, off1, off2], [-2, -1, 0, 1, 2]) / (12 * dx2)
        elif order == 6:
            main = -490 * np.ones(self.nr)
            off1 = 270 * np.ones(self.nr - 1)
            off2 = -27 * np.ones(self.nr - 2)
            off3 = 2 * np.ones(self.nr - 3)
            laplacian = sp.diags([off3, off2, off1, main, off1, off2, off3], [-3, -2, -1, 0, 1, 2, 3]) / (180 * dx2)
        elif order == 8:
            main = -14350 * np.ones(self.nr)
            off1 = 8064 * np.ones(self.nr - 1)
            off2 = -1008 * np.ones(self.nr - 2)
            off3 = 128 * np.ones(self.nr - 3)
            off4 = -9 * np.ones(self.nr - 4)
            laplacian = sp.diags([off4, off3, off2, off1, main, off1, off2, off3, off4], [-4, -3, -2, -1, 0, 1, 2, 3, 4]) / (5040 * dx2)
        else:
            raise ValueError("Only 2nd, 4th, 6th and 8th order finite differences are supported")

        # Kinetic energy operator: T = -1/2 * ∇²
        return -0.5 * laplacian


class Hamiltonian(Potential, Kinetic):
    """Class for building and managing quantum mechanical Hamiltonians."""

    def __init__(self, xmin: float, xmax: float, ngrid: int,
                 potential_type: str, kinetic_method: str = 'fd', fd_order: int = 2):
        Potential.__init__(self, xmin, xmax, ngrid)
        Kinetic.__init__(self, xmin, xmax, ngrid)

        self.potential_type = potential_type
        self.kinetic_method = kinetic_method
        self.fd_order = fd_order
        self._hamiltonian = None

    def set_potential_parameters(self, potential_type: str, params: dict) -> None:
        self.set_potential(potential_type)
        self.set_parameters(params)

    def _build_hamiltonian(self, potential_type: str,
                           params: dict) -> sp.spmatrix:
        # Set potential parameters
        self.set_potential_parameters(potential_type, params)
        # Potential energy term
        V_diagonal = sp.diags(self.evaluate(), 0)

        # Kinetic energy term
        if self.kinetic_method == 'fd':
            T = self.get_finite_diff_kinetic(self.fd_order)
        elif self.kinetic_method == 'dvr':
            T = self.get_dvr_kinetic()
        else:
            raise ValueError("Kinetic method must be 'fd' or 'dvr'")

        return T + V_diagonal

    def build_hamiltonian(self, potential_params: dict) -> sp.spmatrix:
        self._hamiltonian = self._build_hamiltonian(
            self.potential_type, potential_params
        )
        return self._hamiltonian

    @property
    def hamiltonian(self) -> sp.spmatrix:
        """Get the Hamiltonian matrix."""
        if self._hamiltonian is None:
            raise RuntimeError("Hamiltonian not built yet. Call build_hamiltonian() first.")
        return self._hamiltonian

    def eigensolve(self, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
        if self._hamiltonian is None:
            raise RuntimeError("Hamiltonian not built yet. Call build_hamiltonian() first.")

        eigenvalues, eigenvectors = spla.eigsh(self._hamiltonian, k=k, which='SA')
        return eigenvalues, eigenvectors

def test_hamiltonian_orders():
    xmin = -10.0
    xmax = 10.0
    ngrid = 1000
    omega = 1.0
    expected_eigs = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    for order, atol in [(2, 0.1), (4, 0.01), (6, 0.005), (8, 0.001)]:
        h = Hamiltonian(xmin, xmax, ngrid, potential_type='harmonic', kinetic_method='fd', fd_order=order)
        h.build_hamiltonian({'omega': omega})
        eigenvalues, _ = h.eigensolve(k=5)
        np.testing.assert_allclose(eigenvalues, expected_eigs, atol=atol)
        print(f"Test passed for finite difference order {order} with atol={atol}")

    # Test DVR
    h = Hamiltonian(xmin, xmax, ngrid, potential_type='harmonic', kinetic_method='dvr')
    h.build_hamiltonian({'omega': omega})
    eigenvalues, _ = h.eigensolve(k=5)
    np.testing.assert_allclose(eigenvalues, expected_eigs, atol=0.01)
    print("Test passed for DVR")
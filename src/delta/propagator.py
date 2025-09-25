import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


# ----------------------------
# Propagator Base Class
# ----------------------------
class Propagator:
    def __init__(self, hamil, dt):
        self.H = hamil
        self.dt = dt

    def step(self, psi):
        raise NotImplementedError


# ----------------------------
# Crank-Nicolson
# ----------------------------
class CrankNicolson(Propagator):
    def __init__(self, hmat, dt):
        super().__init__(hmat, dt)
        I = sp.identity(hmat.shape[0], format="csr")
        self.A = (I + 0.5j * dt * hmat).tocsc()
        self.B = (I - 0.5j * dt * hmat).tocsc()

    def step(self, psi):
        rhs = self.B.dot(psi)
        return spla.spsolve(self.A, rhs)


# ----------------------------
# Exact Matrix Exponential
# ----------------------------
class MatrixExponential(Propagator):
    def __init__(self, hmat, dt):
        super().__init__(hmat, dt)
        self.U = la.expm(-1j * hmat.toarray() * dt)

    def step(self, psi):
        return self.U @ psi


# ----------------------------
# Split-Operator Method (2nd order Trotter)
# ----------------------------
class SplitOperator(Propagator):
    def __init__(self, hmat, dt, dx):
        super().__init__(hmat, dt)
        self.N = hmat.shape[0]
        self.dx = dx
        self.V = hmat.diagonal()
        k = 2 * np.pi * np.fft.fftfreq(self.N, dx)
        self.T = 0.5 * k ** 2
        self.expV = np.exp(-0.5j * self.dt * self.V)
        self.expT = np.exp(-1j * self.dt * self.T)

    def step(self, psi):
        psi = self.expV * psi
        psi = np.fft.fft(psi)
        psi = self.expT * psi
        psi = np.fft.ifft(psi)
        psi = self.expV * psi
        return psi


# ----------------------------
# Predictor-Corrector (Heun method)
# ----------------------------
class PredictorCorrector(Propagator):
    def step(self, psi):
        k1 = -1j * self.H.dot(psi)
        pred = psi + self.dt * k1
        k2 = -1j * self.H.dot(pred)
        return psi + 0.5 * self.dt * (k1 + k2)


# ----------------------------
# Runge-Kutta 2nd order (midpoint)
# ----------------------------
class RK2(Propagator):
    def step(self, psi):
        k1 = -1j * self.H.dot(psi)
        k2 = -1j * self.H.dot(psi + 0.5 * self.dt * k1)
        return psi + self.dt * k2


# ----------------------------
# Runge-Kutta 4th order
# ----------------------------
class RK4(Propagator):
    def step(self, psi):
        k1 = -1j * self.H.dot(psi)
        k2 = -1j * self.H.dot(psi + 0.5 * self.dt * k1)
        k3 = -1j * self.H.dot(psi + 0.5 * self.dt * k2)
        k4 = -1j * self.H.dot(psi + self.dt * k3)
        return psi + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ----------------------------
# Higher-order Trotter (4th order Yoshida scheme)
# ----------------------------
class HigherOrderTrotter(SplitOperator):
    def step(self, psi):
        # Yoshida coefficients for 4th order
        w1 = 1.0 / (2 - 2**(1/3))
        w2 = -2**(1/3) / (2 - 2**(1/3))

        for w in [w1, w2, w1]:
            expV = np.exp(-0.5j * w * self.dt * self.V)
            expT = np.exp(-1j * w * self.dt * self.T)
            psi = expV * psi
            psi = np.fft.fft(psi)
            psi = expT * psi
            psi = np.fft.ifft(psi)
            psi = expV * psi
        return psi


# ----------------------------
# Lanczos Propagation
# ----------------------------
class LanczosPropagation(Propagator):
    def __init__(self, hmat, dt, krylov_dim=20):
        super().__init__(hmat, dt)
        self.kdim = krylov_dim

    def step(self, psi):
        # Build Krylov subspace using Lanczos
        n = len(psi)
        V = np.zeros((n, self.kdim), dtype=complex)
        T = np.zeros((self.kdim, self.kdim), dtype=complex)
        beta = 0.0
        v = psi / np.linalg.norm(psi)
        V[:, 0] = v
        for j in range(self.kdim - 1):
            w = self.H @ V[:, j]
            alpha = np.vdot(V[:, j], w)
            w = w - alpha * V[:, j] - beta * V[:, j-1] if j > 0 else w - alpha * V[:, j]
            beta = np.linalg.norm(w)
            if beta < 1e-12:
                break
            V[:, j+1] = w / beta
            T[j, j] = alpha
            T[j, j+1] = beta
            T[j+1, j] = beta
        # Exponentiate T
        U_k = la.expm(-1j * self.dt * T)
        return V @ (U_k[:, 0] * np.linalg.norm(psi))


# ----------------------------
# Chebyshev Propagation
# ----------------------------
class ChebyshevPropagation(Propagator):
    def __init__(self, hmat, dt, order=20):
        super().__init__(hmat, dt)
        self.order = order
        # Rescale H to [-1,1]
        evals = la.eigvalsh(hmat.toarray())
        self.a = (np.max(evals) + np.min(evals)) / 2
        self.b = (np.max(evals) - np.min(evals)) / 2
        self.Hs = (hmat - self.a * sp.identity(hmat.shape[0])) / self.b

    def step(self, psi):
        c = [la.jv(k, self.dt * self.b) for k in range(self.order)]  # Bessel coeffs
        phi0 = psi
        phi1 = self.Hs @ psi
        out = c[0] * phi0 + 2j * c[1] * phi1
        phi_prev, phi = phi0, phi1
        for k in range(2, self.order):
            phi_next = 2 * self.Hs @ phi - phi_prev
            out += (2j)**k * c[k] * phi_next
            phi_prev, phi = phi, phi_next
        return np.exp(-1j * self.dt * self.a) * out

def test_propagator(propagator_cls, H, dt, psi0, steps=5, **kwargs):
    """
    Generic test: evolve psi0 and check norm preservation + accuracy
    """
    prop = propagator_cls(H, dt, **kwargs)
    psi = psi0.copy()

    for _ in range(steps):
        psi = prop.step(psi)

    norm = np.linalg.norm(psi)
    assert np.isclose(norm, 1.0, atol=1e-6), f"{propagator_cls.__name__} norm not conserved!"

    # Compare to exact exponential
    U_exact = la.expm(-1j * H.toarray() * dt * steps)
    psi_exact = U_exact @ psi0
    err = np.linalg.norm(psi - psi_exact)
    print(f"{propagator_cls.__name__}: norm={norm:.6f}, err={err:.2e}")
    return err


def run_all_tests():
    np.random.seed(42)
    N = 6
    dt = 0.05
    steps = 5

    # Hermitian Hamiltonian
    H_dense = np.random.rand(N, N) + 1j*np.random.rand(N, N)
    H_dense = 0.5 * (H_dense + H_dense.conj().T)
    H = sp.csr_matrix(H_dense)

    psi0 = np.random.rand(N) + 1j*np.random.rand(N)
    psi0 /= np.linalg.norm(psi0)

    errors = {}
    errors["CN"] = test_propagator(CrankNicolson, H, dt, psi0, steps)
    errors["Exp"] = test_propagator(MatrixExponential, H, dt, psi0, steps)
    errors["PredictorCorrector"] = test_propagator(PredictorCorrector, H, dt, psi0, steps)
    errors["RK2"] = test_propagator(RK2, H, dt, psi0, steps)
    errors["RK4"] = test_propagator(RK4, H, dt, psi0, steps)
    errors["Lanczos"] = test_propagator(LanczosPropagation, H, dt, psi0, steps, krylov_dim=4)
    errors["Chebyshev"] = test_propagator(ChebyshevPropagation, H, dt, psi0, steps, order=10)

    # Split operator / Trotter require diagonal + FFT grid
    Ngrid = 64
    dx = 0.1
    x = np.linspace(-Ngrid//2, Ngrid//2, Ngrid) * dx
    V = 0.5 * x**2
    H_diag = sp.diags(V, format="csr")
    psi0_grid = np.exp(-x**2)  # Gaussian
    psi0_grid = psi0_grid / np.linalg.norm(psi0_grid)

    errors["SplitOp"] = test_propagator(SplitOperator, H_diag, dt, psi0_grid, steps, dx=dx)
    errors["HighOrderTrotter"] = test_propagator(HigherOrderTrotter, H_diag, dt, psi0_grid, steps, dx=dx)

    print("\nSummary of L2 errors vs exact exponential:")
    for k, v in errors.items():
        print(f"{k:20s} error = {v:.2e}")
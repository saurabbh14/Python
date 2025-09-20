import math
import numpy as np
from scipy.integrate import quad

# Peice wise delta function for the potential
class DeltaFuncPoten:
    def __init__(self, V0=8.0, alpha=20.0, potential_type='exponential'):
        self.V0 = V0
        self.alpha = alpha
        self.potential_type = potential_type

    def V(self, x):
        if self.potential_type == 'exponential':
            return self.V0 * math.exp(-self.alpha * abs(x))
        # Add other types as needed, e.g.:
        # elif self.potential_type == 'gaussian':
        #     return 2 * math.sqrt(100 / math.pi) * math.exp(-100 * x**2)
        raise ValueError("Unsupported potential type")

    def discretize(self, dx=0.07, threshold=0.001):
        if self.potential_type == 'exponential':
            x0 = math.log(self.V0 / threshold) / self.alpha
        # Add for other types as needed
        else:
            raise ValueError("Unsupported potential type for discretization")
        self.a0 = -x0
        self.an = x0
        num_deltas = math.floor((self.an - self.a0) / dx)
        self.n = num_deltas + 1
        self.dx = dx
        self.a = [self.a0 + i * self.dx for i in range(self.n + 1)]
        self.c = []
        for i in range(1, self.n + 1):
            integral, _ = quad(self.V, self.a[i-1], self.a[i])
            self.c.append(integral)

# Scattering equation solver for any type of potential
class ScatteringSolver:
    def __init__(self, Vfunc, hbar=1.0, m=1.0, dx=0.07, threshold=1e-7, L=10.0):
        """
        Vfunc: potential class called for the scattering
        """
        self.Vfunc = Vfunc
        self.hbar = hbar
        self.m = m
        self.dx = dx
        self.threshold = threshold

        self.a0, self.an = self.find_support(L=L)
        self.nodes, self.c_vals = self.compute_deltas()

    def find_support(self, L=10.0, N=20001):
        xs = np.linspace(-L, L, N)
        vals = self.Vfunc(xs)
        mask = vals >= self.threshold
        if not np.any(mask):
            return -1.0, 1.0
        return float(xs[mask][0]), float(xs[mask][-1])

    def compute_deltas(self):
        Ltot = self.an - self.a0
        n_seg = int(math.ceil(Ltot / self.dx))
        nodes = self.a0 + np.arange(n_seg+1)*self.dx
        c_vals = np.empty(n_seg, dtype=float)
        for i in range(n_seg):
            xL, xR = nodes[i], nodes[i+1]
            c_vals[i] = 0.5*(self.Vfunc(xL) + self.Vfunc(xR))*(xR-xL)
        return nodes, c_vals

    def A(self, a, V_region, E):
        k = np.sqrt(2*self.m*(E - V_region) + 0j)/self.hbar
        exp_ika, exp_mika = np.exp(1j*k*a), np.exp(-1j*k*a)
        return np.array([[exp_ika, exp_mika],
                         [-1j*k*exp_ika, 1j*k*exp_mika]], dtype=complex)

    def Em(self, a, V_region, c, E):
        k = np.sqrt(2 * self.m * (E - V_region) + 0j)/self.hbar
        exp_ika, exp_mika = np.exp(1j*k*a), np.exp(-1j*k*a)
        term = (2 * self.m / self.hbar**2)*c
        return np.array([[exp_ika, exp_mika],
                         [(-1j * k + term) * exp_ika, (1j * k + term) * exp_mika]], dtype=complex)

    def transfer_matrix(self, E):
        T = np.eye(2, dtype=complex)
        for o, c in enumerate(self.c_vals):
            a_boundary = self.nodes[o + 1]
            A_mat  = self.A(a_boundary, 0.0, E)
            Em_mat = self.Em(a_boundary, 0.0, c, E)
            step = np.linalg.solve(A_mat, Em_mat)
            T = T @ step
        return T

    def transmission_reflection(self, E_vals):
        T_vals, R_vals = [], []
        for E in E_vals:
            Tmat = self.transfer_matrix(E)
            denom = abs(Tmat[0,0])**2
            if denom == 0:
                T_vals.append(np.nan)
                R_vals.append(np.nan)
            else:
                T_vals.append(1.0/denom)
                R_vals.append(abs(Tmat[1,0]/Tmat[0,0])**2)
        return np.array(T_vals), np.array(R_vals)

# Check the difference and finalize
class Scattering:
    def __init__(self, potential, hbar=1.0, m=1.0, omega_L=0.01):
        self.potential = potential
        self.hbar = hbar
        self.m = m
        self.omega_L = omega_L

    def compute_tau_z_table(self, en_start=0.00001, en_end=None, steps=21):
        if en_end is None:
            en_end = 2 * self.potential.V0
        ens = np.linspace(en_start, en_end, steps)
        table = []
        for En in ens:
            if En <= 0:
                continue  # Skip non-positive energies
            k = np.sqrt(2 * self.m * En) / self.hbar
            # Build Tnplus and Tnminus in reverse order for correct multiplication (right to left)
            Tnplus = np.eye(2, dtype=complex)
            Tnminus = np.eye(2, dtype=complex)
            for i in range(self.potential.n, 0, -1):  # Reverse: n down to 1
                pos = self.potential.a[i]
                exp_pos = np.exp(1j * k * pos)
                exp_neg = np.exp(-1j * k * pos)
                A = np.array([[exp_pos, exp_neg],
                              [-1j * k * exp_pos, 1j * k * exp_neg]], dtype=complex)
                inv_A = np.linalg.inv(A)
                # Plus
                cp = self.potential.c[i-1] + self.hbar * self.omega_L / 2
                alpha_p = 2 * self.m * cp / self.hbar**2
                Em_p = np.array([[exp_pos, exp_neg],
                                 [-1j * k * exp_pos + alpha_p * exp_pos,
                                  1j * k * exp_neg + alpha_p * exp_neg]], dtype=complex)
                Mc_p = inv_A @ Em_p
                Tnplus = Tnplus @ Mc_p
                # Minus
                cm = self.potential.c[i-1] - self.hbar * self.omega_L / 2
                alpha_m = 2 * self.m * cm / self.hbar**2
                Em_m = np.array([[exp_pos, exp_neg],
                                 [-1j * k * exp_pos + alpha_m * exp_pos,
                                  1j * k * exp_neg + alpha_m * exp_neg]], dtype=complex)
                Mc_m = inv_A @ Em_m
                Tnminus = Tnminus @ Mc_m
            tp = 1 / np.abs(Tnplus[0, 0])**2
            tm = 1 / np.abs(Tnminus[0, 0])**2
            tau = (1 / self.omega_L) * (tp - tm) / (tp + tm)
            table.append((En, tau))
        return table
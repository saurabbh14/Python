import numpy as np
from scipy.sparse import spdiags, eye
from scipy.linalg import expm
from scipy.fft import ifft
import matplotlib.pyplot as plt

from potential.grid import TGrid
from potential.gridFunc import Potential, WavePacket


# One-dimensional wave packet propagation using finite-difference method
class WavePacketPropagation(Potential, WavePacket):
    def __init__(self, xmin, xmax, ngrid):
        Potential.__init__(self, xmin, xmax, ngrid)
        WavePacket.__init__(self, xmin, xmax, ngrid)
        self.tgrid = None
        self.psi = None

    def phi(self)->np.ndarray:
        a = 20
        # Make psi(x,0) from Gaussian kspace wavefunction phi(k) using fast fourier transform
        phi = np.exp(-a * (self.k_grid - self.p0)**2) * np.exp(-1j * 6 * self.k_grid**2)  # unnormalized phi(k)
        return phi

    def psi0(self):
        phi = self.phi()
        psi = ifft(phi)
        psi /= np.sqrt(np.dot(self.psi.conj().T, self.psi) * self.dR)  # normalize the psi(x,0)
        self.psi = psi

    def avgE(self):
        phi = self.phi()
        # Expectation value of energy; e.g. for the parameters chosen above <E> = 2.062.
        energy = np.dot(phi.conj().T, 0.5 * self.k_grid**2 * phi) * self.dK / (np.dot(phi.conj().T, phi) * self.dK)
        return energy

    def potential(self, x):
        # CHOOSE POTENTIAL U(X): Either U = 0 OR U = step potential of height avgE that is located at x=L/2
        # self.U = np.zeros_like(self.x)  # free particle wave packet evolution
        self.set_potential('StepFunc')
        self.set_parameters(self.avgE())
        self.U = self.evaluate()

    def setupHamiltonian(self):
        # Finite-difference representation of Laplacian and Hamiltonian
        e = np.ones(self.nr)
        data = np.array([e, -2*e, e])
        diags = np.array([-1, 0, 1])
        Lap = spdiags(data, diags, self.nr, self.nr) / self.dR**2
        self.H = -0.5 * Lap + spdiags([self.U], [0], self.nr, self.nr)

    def setup_time(self):
        tgrid = TGrid(0., 20, 200)
        self.tgrid = tgrid

    def microstep(self):
        # Time displacement operator E = exp(-i H dT / hbar)
        self.E = expm(-1j * self.H.toarray() * self.dT / self.hbar)  # Using full matrix for expm

    def simulate(self):
        global rho
        plt.ion()  # Interactive mode for live plotting
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, np.zeros_like(self.x), 'k')
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, 0.15)
        ax.set_xlabel('x [m]', fontsize=24)
        ax.set_ylabel('probability density [1/m]', fontsize=24)

        for t in range(self.NT):
            # Calculate new psi from old psi
            self.psi = np.dot(self.E, self.psi)
            # Calculate probability density rho(x,T)
            rho = np.conj(self.psi) * self.psi

            # Update plot
            line.set_ydata(rho)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)

        plt.ioff()
        plt.show()

        # Calculate Reflection probability
        R = 0
        for a in range(self.N // 2):
            R += rho[a]
        R *= self.dx
        return R

def test_1Dim_wavepacket():
    sim = WavePacketPropagation()
    reflection_prob = sim.simulate()
    print(f"Reflection Probability R: {reflection_prob}")
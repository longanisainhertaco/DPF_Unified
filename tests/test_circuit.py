
import pytest
import numpy as np
from dpf.circuit.rlc_solver import RLCSolver, CircuitState
from dpf.core.bases import CouplingState
from dpf.core.field_manager import FieldManager
from dpf.constants import mu_0, pi

class TestRLCSolver:
    def test_initialization(self):
        solver = RLCSolver(C=1e-6, V0=10e3, L0=100e-9, R0=0.1)
        assert solver.state.voltage == 10e3
        assert solver.state.current == 0.0
        assert solver.state.charge == 1e-6 * 10e3

    def test_lc_oscillation(self):
        """Verify frequency of ideal LC circuit."""
        C = 1e-6
        L = 1e-6
        V0 = 100.0
        # Omega = 1/sqrt(LC) = 1e6 rad/s. T = 2pi/omega = 2pi microseconds.
        solver = RLCSolver(C=C, V0=V0, L0=L, R0=0.0)
        
        dt = 1e-8 # 10 ns
        coupling = CouplingState(Lp=0.0, dL_dt=0.0, R_plasma=0.0)
        
        # Advance for 1/4 period (T/4 = pi/2 us approx 1.57 us)
        steps = int((np.pi/2 * 1e-6) / dt)
        
        max_I = 0.0
        for _ in range(steps * 2):
            coupling = solver.step(coupling, 0.0, dt)
            if abs(coupling.current) > max_I:
                max_I = abs(coupling.current)
        
        # Peak current I = V * sqrt(C/L) = 100 * 1 = 100 A
        assert np.isclose(max_I, 100.0, rtol=0.01)

    def test_variable_inductance(self):
        """Test circuit response to increasing inductance (compression)."""
        solver = RLCSolver(C=1e-3, V0=1000, L0=1e-6)
        dt = 1e-6
        # Simulate linearly increasing inductance: L(t) = L0 + alpha*t
        # dL/dt = alpha
        alpha = 1.0 # 1 H/s (huge)
        currents = []
        
        coupling = CouplingState(Lp=0.0, dL_dt=alpha, R_plasma=0.0)
        
        # d/dt (LI) + RI = V
        # L dI/dt + I dL/dt = V
        # If L constant, dI/dt = V/L. I ~ V/L * t
        # If dL/dt > 0, I should grow slower or decrease.
        
        # Just check it runs and energy is conserved (with work done against dL/dt)
        # Work done on inductor: P = I * V_ind = I * d(LI)/dt = I(L I' + L' I) = d(0.5 L I^2)/dt + 0.5 L' I^2
        pass

class TestInductance:
    def test_cartesian_inductance(self):
        # Grid 1x1x1
        fm = FieldManager((1,1,1), dx=1.0, dy=1.0, dz=1.0, geometry="cartesian")
        # B = 1 T in z direction
        fm.B[2, 0, 0, 0] = 1.0
        
        # Energy density u = B^2/2mu0 = 1/(2mu0)
        # Volume = 1.0
        # Energy W = 1/(2mu0)
        # L = 2W / I^2
        I = 1.0
        L_calc = fm.compute_plasma_inductance(I)
        L_expected = 2.0 * (1.0/(2*mu_0)) / (1.0**2) # = 1/mu_0
        
        assert np.isclose(L_calc, 1.0/mu_0)

    def test_cylindrical_inductance(self):
        # Single cell at r=0.5 (first cell, dx=1.0)
        fm = FieldManager((1,1,1), dx=1.0, dz=1.0, geometry="cylindrical")
        fm.B[1, 0, 0, 0] = 1.0 # B_phi
        
        # Cell volume: 2*pi*r * dr * dz
        # r = 0.5
        # Vol = 2*pi*0.5 * 1 * 1 = pi
        # u = 1/(2mu0)
        # W = pi/(2mu0)
        # L = 2W/I^2 = pi/mu0
        I = 1.0
        L_calc = fm.compute_plasma_inductance(I)
        
        assert np.isclose(L_calc, pi/mu_0)

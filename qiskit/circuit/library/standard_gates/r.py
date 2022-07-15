# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around an axis in x-y plane."""

import math
from cmath import exp
from typing import Optional
import numpy
from qiskit.qasm import pi
from qiskit.circuit import _instruction_parameter_shims as _shims
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType


class RGate(Gate):
    r"""Rotation θ around the cos(φ)x + sin(φ)y axis.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────┐
        q_0: ┤ R(ϴ) ├
             └──────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R(\theta, \phi) = e^{-i \th \left(\cos{\phi} x + \sin{\phi} y\right)} =
            \begin{pmatrix}
                \cos{\th} & -i e^{-i \phi} \sin{\th} \\
                -i e^{i \phi} \sin{\th} & \cos{\th}
            \end{pmatrix}
    """

    _spec = (_shims.FloatType(),) * 2

    def __init__(
        self,
        theta: Optional[ParameterValueType] = None,
        phi: Optional[ParameterValueType] = None,
        label: Optional[str] = None,
    ):
        """Create new r single-qubit gate."""
        parameters = None if theta is None else [theta, phi]
        super().__init__("r", 1, parameters, label=label, _shim_parameter_spec=self._spec)

    def _decompose(self, parameters):
        """
        gate r(θ, φ) a {u3(θ, φ - π/2, -φ + π/2) a;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        theta, phi = parameters
        qc = QuantumCircuit(1, name=self.name)
        qc.u(theta, phi - pi / 2, -phi + pi / 2, 0)
        return qc

    def inverse(self):
        """Invert this gate.

        r(θ, φ)^dagger = r(-θ, φ)
        """
        return RGate(-self.params[0], self.params[1])

    def __array__(self, dtype=None):
        """Return a numpy.array for the R gate."""
        theta, phi = float(self.params[0]), float(self.params[1])
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = exp(-1j * phi)
        exp_p = exp(1j * phi)
        return numpy.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]], dtype=dtype)

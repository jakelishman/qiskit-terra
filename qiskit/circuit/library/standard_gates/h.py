# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hadamard gate."""
from math import sqrt
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit import CircuitInstruction
from qiskit.qasm import pi


class HGate(Gate):
    r"""Single-qubit Hadamard gate.

    This gate is a \pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\rangle,|1\rangle` to
    :math:`|+\rangle,|-\rangle` and vice-versa.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}
    """

    def __init__(self, label: Optional[str] = None):
        """Create new H gate."""
        super().__init__("h", 1, label=label, _shim_parameter_spec=())

    def _decompose(self, parameters):
        """
        gate h a { u2(0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        qc = QuantumCircuit(1, name=self.name)
        qc.u(0, 0, pi, 0)
        return qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
    ):
        """Return a (multi-)controlled-H gate.

        One control qubit returns a CH gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CHGate(label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted H gate (itself)."""
        return HGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a Numpy.array for the H gate."""
        return numpy.array([[1, 1], [1, -1]], dtype=dtype) / numpy.sqrt(2)


class CHGate(ControlledGate):
    r"""Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\rangle` state.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        CH\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + H \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
                0 & 0 & 1 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ H ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CH\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes H =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
                    0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
                \end{pmatrix}
    """
    # Define class constants. This saves future allocation time.
    _sqrt2o2 = 1 / sqrt(2)
    _matrix1 = numpy.array(
        [[1, 0, 0, 0], [0, _sqrt2o2, 0, _sqrt2o2], [0, 0, 1, 0], [0, _sqrt2o2, 0, -_sqrt2o2]],
        dtype=complex,
    )
    _matrix0 = numpy.array(
        [[_sqrt2o2, 0, _sqrt2o2, 0], [0, 1, 0, 0], [_sqrt2o2, 0, -_sqrt2o2, 0], [0, 0, 0, 1]],
        dtype=complex,
    )

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[int, str]] = None):
        """Create new CH gate."""
        super().__init__(
            "ch",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=HGate(),
            _shim_parameter_spec=(),
        )

    def _decompose(self, parameters):
        """
        gate ch a,b {
            s b;
            h b;
            t b;
            cx a, b;
            tdg b;
            h b;
            sdg b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import SGate, SdgGate, TGate, TdgGate, CXGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(CircuitInstruction(SGate(), (q[1],), (), ()))
        qc._append(CircuitInstruction(HGate(), (q[1],), (), ()))
        qc._append(CircuitInstruction(TGate(), (q[1],), (), ()))
        qc._append(CircuitInstruction(CXGate(), (q[0], q[1]), (), ()))
        qc._append(CircuitInstruction(TdgGate(), (q[1],), (), ()))
        qc._append(CircuitInstruction(HGate(), (q[1],), (), ()))
        qc._append(CircuitInstruction(SdgGate(), (q[1],), (), ()))
        return qc

    def inverse(self):
        """Return inverted CH gate (itself)."""
        return CHGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the CH gate."""
        mat = self._matrix1 if self.ctrl_state else self._matrix0
        if dtype:
            return numpy.asarray(mat, dtype=dtype)
        return mat

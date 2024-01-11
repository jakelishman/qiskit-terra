// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple, PyType};

use crate::error::QASM3ImporterError;

macro_rules! register_type {
    ($name: ident) => {
        /// Rust-space wrapper around Qiskit `Register` objects.
        pub struct $name {
            /// The actual register instance.
            object: Py<PyAny>,
            /// A pointer to the inner list of bits.  We keep a handle to this for lookup
            /// efficiency; we can use direct list methods to retrieve the bit instances, rather
            /// than needing to indirect through the general `__getitem__` of the register, or
            /// looking up the qubit instances on the circuit.
            items: Py<PyList>,
        }

        impl $name {
            /// Get an individual bit from the register.
            pub fn bit(&self, py: Python, index: usize) -> PyResult<Py<PyAny>> {
                // Unfortunately, `PyList::get_item_unchecked` isn't usable with the stable ABI.
                self.items
                    .as_ref(py)
                    .get_item(index)
                    .map(|item| item.into_py(py))
            }

            pub fn iter<'a>(&'a self, py: Python<'a>) -> impl Iterator<Item = &'a PyAny> {
                self.items.as_ref(py).iter()
            }
        }

        impl ::pyo3::IntoPy<Py<PyAny>> for $name {
            fn into_py(self, _py: Python) -> Py<PyAny> {
                self.object
            }
        }

        impl ::pyo3::ToPyObject for $name {
            fn to_object(&self, _py: Python) -> Py<PyAny> {
                // _Technically_, allowing access this internal object can let the Rust-space
                // wrapper get out-of-sync since we keep a direct handle to the list, but in
                // practice, the field it's viewing is private and "inaccessible" from Python.
                self.object.clone()
            }
        }
    };
}

register_type!(PyQuantumRegister);
register_type!(PyClassicalRegister);

/// Information received from Python space about how to construct a Python-space object to
/// represent a given gate that might be declared.
#[pyclass(module = "qiskit._qasm3", frozen, name = "CustomGate")]
#[derive(Clone, Debug)]
pub struct PyGate {
    constructor: Py<PyAny>,
    name: String,
    num_params: usize,
    num_qubits: usize,
}

impl PyGate {
    pub fn new<T: IntoPy<Py<PyAny>>>(
        py: Python,
        constructor: T,
        name: String,
        num_params: usize,
        num_qubits: usize,
    ) -> Self {
        Self {
            constructor: constructor.into_py(py),
            name,
            num_params,
            num_qubits,
        }
    }

    /// Construct a Python-space instance of the custom gate.
    pub fn construct<A>(&self, py: Python, args: A) -> PyResult<Py<PyAny>>
    where
        A: IntoPy<Py<PyTuple>>,
    {
        let args = args.into_py(py);
        let received_num_params = args.as_ref(py).len();
        if received_num_params == self.num_params {
            self.constructor.call1(py, args.as_ref(py))
        } else {
            Err(QASM3ImporterError::new_err(format!(
                "internal logic error: wrong number of params for {} (got {}, expected {})",
                &self.name, received_num_params, self.num_params
            )))
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[pymethods]
impl PyGate {
    #[new]
    #[pyo3(signature=(/, constructor, name, num_params, num_qubits))]
    fn __new__(constructor: Py<PyAny>, name: String, num_params: usize, num_qubits: usize) -> Self {
        Self {
            constructor,
            name,
            num_params,
            num_qubits,
        }
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        PyString::new(py, "CustomGate(name={!r}, num_params={}, num_qubits={})").call_method1(
            "format",
            (
                PyString::new(py, &self.name),
                self.num_params,
                self.num_qubits,
            ),
        )
    }

    fn __reduce__(&self, py: Python) -> Py<PyTuple> {
        (
            PyType::new::<PyGate>(py),
            (
                self.constructor.clone_ref(py),
                &self.name,
                self.num_params,
                self.num_qubits,
            ),
        )
            .into_py(py)
    }
}

/// Wrapper around various Python-space imports. This is just a convenience wrapper to save us
/// needing to `getattr` things off a Python-space module quite so frequently.  This is
/// give-or-take just a manual lookup for a few `import` items at the top of a Python module, and
/// the attached constructor functions produce (when appropriate), Rust-space wrappers around the
/// Python objects.
pub struct PyCircuitModule {
    circuit: Py<PyType>,
    qreg: Py<PyType>,
    qubit: Py<PyType>,
    creg: Py<PyType>,
    clbit: Py<PyType>,
    instruction: Py<PyType>,
}

impl PyCircuitModule {
    /// Import the necessary components from `qiskit.circuit`.
    pub fn import(py: Python) -> PyResult<Self> {
        let module = PyModule::import(py, "qiskit.circuit")?;
        Ok(Self {
            circuit: module
                .getattr("QuantumCircuit")?
                .downcast::<PyType>()?
                .into_py(py),
            qreg: module
                .getattr("QuantumRegister")?
                .downcast::<PyType>()?
                .into_py(py),
            qubit: module.getattr("Qubit")?.downcast::<PyType>()?.into_py(py),
            creg: module
                .getattr("ClassicalRegister")?
                .downcast::<PyType>()?
                .into_py(py),
            clbit: module.getattr("Clbit")?.downcast::<PyType>()?.into_py(py),
            instruction: module
                .getattr("CircuitInstruction")?
                .downcast::<PyType>()?
                .into_py(py),
        })
    }

    pub fn new_circuit(&self, py: Python) -> PyResult<PyCircuit> {
        Ok(PyCircuit {
            qc: self.circuit.call0(py)?,
        })
    }

    pub fn new_qreg<T: IntoPy<Py<PyString>>>(
        &self,
        py: Python,
        name: T,
        size: usize,
    ) -> PyResult<PyQuantumRegister> {
        let qreg = self.qreg.call1(py, (size, name.into_py(py)))?;
        Ok(PyQuantumRegister {
            items: qreg
                .getattr(py, "_bits")?
                .downcast::<PyList>(py)?
                .into_py(py),
            object: qreg,
        })
    }

    pub fn new_qubit(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.qubit.call0(py)
    }

    pub fn new_creg<T: IntoPy<Py<PyString>>>(
        &self,
        py: Python,
        name: T,
        size: usize,
    ) -> PyResult<PyClassicalRegister> {
        let creg = self.creg.call1(py, (size, name.into_py(py)))?;
        Ok(PyClassicalRegister {
            items: creg
                .getattr(py, "_bits")?
                .downcast::<PyList>(py)?
                .into_py(py),
            object: creg,
        })
    }

    pub fn new_clbit(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.clbit.call0(py)
    }

    pub fn new_instruction<O, Q, C>(
        &self,
        py: Python,
        operation: O,
        qubits: Q,
        clbits: C,
    ) -> PyResult<Py<PyAny>>
    where
        O: IntoPy<Py<PyAny>>,
        Q: IntoPy<Py<PyTuple>>,
        C: IntoPy<Py<PyTuple>>,
    {
        self.instruction
            .call1(py, (operation, qubits.into_py(py), clbits.into_py(py)))
    }
}

/// Circuit construction context object to provide an easier Rust-space interface for us to
/// construct the Python :class:`.QuantumCircuit`.  The idea of doing this from Rust space like
/// this is that we might steadily be able to move more and more of it into being native Rust as
/// the Rust-space APIs around the internal circuit data stabilise.
pub struct PyCircuit {
    /// The actual circuit object that's under construction.
    qc: Py<PyAny>,
}

impl PyCircuit {
    pub fn add_qreg(&mut self, py: Python, qreg: &PyQuantumRegister) -> PyResult<()> {
        self.qc
            .call_method1(py, "add_register", (qreg.to_object(py),))
            .map(|_| ())
    }

    pub fn add_qubit(&mut self, py: Python, qubit: Py<PyAny>) -> PyResult<()> {
        self.qc
            .call_method1(py, "add_bits", ((qubit,),))
            .map(|_| ())
    }

    pub fn add_creg(&mut self, py: Python, creg: &PyClassicalRegister) -> PyResult<()> {
        self.qc
            .call_method1(py, "add_register", (creg.to_object(py),))
            .map(|_| ())
    }

    pub fn add_clbit<T: IntoPy<Py<PyAny>>>(&mut self, py: Python, clbit: T) -> PyResult<()> {
        self.qc
            .call_method1(py, "add_bits", ((clbit,),))
            .map(|_| ())
    }

    pub fn append<T: IntoPy<Py<PyAny>>>(&mut self, py: Python, instruction: T) -> PyResult<()> {
        self.qc
            .call_method1(py, "_append", (instruction.into_py(py),))
            .map(|_| ())
    }
}

impl ::pyo3::IntoPy<Py<PyAny>> for PyCircuit {
    fn into_py(self, py: Python) -> Py<PyAny> {
        self.qc.clone_ref(py)
    }
}

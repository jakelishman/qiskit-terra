# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import abc
import collections.abc
import functools
import weakref

from typing import List, Any

import numpy

from .parameterexpression import ParameterExpression


class ParameterType(abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    def dynamic(self):
        """Whether this type can be part of a dynamic circuit (``True``), or if it represents some
        circuit-state state of the object that operates on it (``False``)."""

    @abc.abstractmethod
    def bind(self, value):
        """Type check the given ``value``, returning it in a normalised form if it is valid for this
        type, or raising ``TypeError`` if it is not.

        Raises:
            TypeError: if the given value is not a suitable type.
        """


class OpaqueType(ParameterType):
    __slots__ = ()

    dynamic = False

    def bind(self, value):
        return value


class FloatType(ParameterType):
    __slots__ = ()

    dynamic = True

    def bind(self, value):
        if isinstance(value, (float, numpy.floating, ParameterExpression)):
            return value
        if isinstance(value, (int, numpy.integer)):
            return float(value)
        raise TypeError(f"required a floating-point number, but received '{value}'")


class ParameterBackreferences(collections.abc.MutableSequence):
    """A sequence-like object to handle backwards compatibility as parameters are moved out of
    :class:`~.circuit.Instruction` instances and into the circuit-level
    :class:`.CircuitInstruction`.

    This should only be needed during the transition period, and is intended to be removed once the
    only method of accessing circuit-runtime-mutable parameters is via
    :attr:`.CircuitInstruction.parameters` rather than :attr:`Instruction.params`.

    :class:`.QuantumCircuit` should ensure that any instruction that actually has parameters cannot
    be added to the circuit without a copy being made.  This means that for any valid usage of
    :class:`.QuantumCircuit` before parameters were moved, a user accessing
    :attr:`Instruction.params <.circuit.Instruction.params>` for an instruction that is in a circuit
    (i.e. bound, in the language of this class) can only have one reference within a circuit.  This
    makes it safe for us to assume that those accessing this method of this object will have only
    bound the referenced :class:`~.circuit.Instruction` in one place.

    The logic is a little complicated here, because we have to have a place to store parameters that
    are set on an :class:`~.circuit.Instruction` instances, so they can be transferred to the
    :class:`.CircuitInstruction` when the object is added to a circuit.
    """

    __slots__ = (
        "_bound_reference",
        "_local_parameters",
        "_local_keys",
        "_foreign_key_map",
        "_getters",
        "_setters",
    )

    __ABSENT = object()
    """Sentinel value for parameters that have not been stored in this object.  In new-style usage,
    this should be _all_ non-state parameters."""

    def __init__(self, parameter_spec):
        self._bound_reference = None
        """A weak reference to the :class:`.CircuitInstruction` that contains the foreign parameters
        for this object.  The outer circuit instruction should contain the :class:`Instruction` that
        this object is bound to.  We store only a weak reference, because we don't want to
        inadvertantly keep the containing object alive if our :class:`Instruction` is removed from
        it; this could easily leak memory, and stress out the garbage collector with many circular
        references."""
        self._local_parameters = [self.__ABSENT] * len(parameter_spec)
        """Storage for local parameters.  This is effectively the replacement of
        `Instruction._params` from before this class existed."""
        self._foreign_key_map = {}
        """Mapping of ``{local_index: foreign_index}``; if an index is a key in this map, then the
        getter should instead look to the foreign reference, and retrieve index ``foreign_index``
        from it instead, if the reference has been bound."""
        self._local_keys = []
        """The keys that are explicitly always local.  These are the "state-like" params."""
        foreign_key = 0
        for local_key, type_ in enumerate(parameter_spec):
            if type_.dynamic:
                self._foreign_key_map[local_key] = foreign_key
                foreign_key += 1
            else:
                self._local_keys.append(local_key)
        # Before we're bound to a containing object, all getters and setters point to the local
        # storage.
        self._getters = [functools.partial(self._get_local_key, i) for i in range(len(self))]
        self._setters = [functools.partial(self._set_local_key, i) for i in range(len(self))]

    def __len__(self):
        return len(self._local_parameters)

    def __iter__(self):
        for getter in self._getters:
            yield getter()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._getters[i]() for i in range(*key.indices(len(self)))]
        return self._getters[key]()

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            values = tuple(value)
            for value_index, our_index in enumerate(range(*key.indices(len(self)))):
                self._setters[our_index](values[value_index])
            return
        self._setters[key](value)

    def copy(self):
        """Return a shallow copy of these parameters."""
        # Shallowness in this context also means shallow-copying the internal mutable objects, but
        # not deep-copying them. This avoids a copied set of parameters holding the same references.
        out = type(self).__new__(type(self))
        local_parameters, local_keys, foreign_key_map = self.__getstate__()
        out.__setstate__((local_parameters.copy(), local_keys.copy(), foreign_key_map.copy()))
        return out

    def __getstate__(self):
        # We must discard the un-pickleable weakref; we will gain it back when the containing
        # `CircuitInstruction` reconstructs itself.
        return (self._local_parameters, self._local_keys, self._foreign_key_map)

    def __setstate__(self, state):
        self._bound_reference = None
        self._local_parameters = state[0]
        self._local_keys = state[1]
        self._foreign_key_map = state[2]
        self._getters = [functools.partial(self._get_local_key, i) for i in range(len(self))]
        self._setters = [functools.partial(self._set_local_key, i) for i in range(len(self))]

    __copy__ = copy
    # __deepcopy__ uses a recursive pickle by default, which will work just fine for us.

    def __delitem__(self, _key):
        raise NotImplementedError("cannot change the number of parameters in an instruction")

    def insert(self, _key, _value):
        raise NotImplementedError("cannot change the number of parameters in an instruction")

    def reference(self, circuit_instruction):
        """Bind this parameter store to the given :class:`.CircuitInstruction`.

        Any run-time parameters that are present, but ought to have been stored in the
        :class:`.CircuitInstruction` instead will be transferred to the new binding.  Any necessary
        getters and setters from this class will be updated to instead read and write to the foreign
        location.

        This needs to be a separately called method to support existing code that might do, for
        example, ``circuit.append(RZGate(np.pi), [0], [])`` with the object constructed with
        parameters.  We need the regular parameter-setting machinery to continue working until a
        circuit takes ownership of the instruction."""
        self._bound_reference = weakref.ref(circuit_instruction)
        for local_key, foreign_key in self._foreign_key_map.items():
            self._getters[local_key] = functools.partial(self._get_foreign_key, foreign_key)
            self._setters[local_key] = functools.partial(self._set_foreign_key, foreign_key)

    def dynamic_parameters(self) -> List[Any]:
        """A list of the parameters that are considered "dynamic", *i.e.* those that could in theory
        be set during a circuit execution by the executing hardware."""
        return [self._getters[i]() for i in self._foreign_key_map]

    def state_parameters(self) -> List[Any]:
        """A list of the parameters that are considered the "state" of the object, *i.e.* those that
        are not settable dynamically during circuit execution."""
        return [self._getters[i]() for i in self._local_keys]

    def _get_local_key(self, key: int):
        out = self._local_parameters[key]
        if out is self.__ABSENT:
            raise KeyError(f"parameter {key} is not set")
        return out

    def _set_local_key(self, key: int, value):
        self._local_parameters[key] = value

    def _get_foreign_key(self, key: int):
        return self._bound_reference().parameters[key]

    def _set_foreign_key(self, key: int, value):
        self._bound_reference().parameters[key] = value

    def __repr__(self):
        return repr(list(self))

    def __str__(self):
        return str(list(self))

    def __eq__(self, other):
        return list(self) == other

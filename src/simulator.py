from backend import Backend
from circuit import Circuit
from gate_fusion import fuse_circuit
from statevector import StateVector


class StateVectorSimulator:
    def __init__(
        self,
        backend: Backend | None = None,
        *,
        fusion: bool = False,
    ) -> None:
        if not isinstance(fusion, bool):
            raise TypeError("fusion must be a boolean")

        self._backend = backend
        self._fusion = fusion

    @property
    def backend(self) -> Backend | None:
        return self._backend

    @property
    def fusion(self) -> bool:
        return self._fusion

    def run(
        self,
        circuit: Circuit,
        initial_state: StateVector | None = None,
    ) -> StateVector:
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a Circuit")

        if initial_state is None:
            state = StateVector(circuit.num_qubits, backend=self._backend)
        else:
            if initial_state.num_qubits != circuit.num_qubits:
                raise ValueError(
                    f"circuit acts on {circuit.num_qubits} qubits, "
                    f"but the initial state has {initial_state.num_qubits}"
                )
            state = StateVector(
                circuit.num_qubits,
                initial_state.raw_amplitudes,
                backend=(
                    self._backend
                    if self._backend is not None
                    else initial_state.backend
                ),
            )

        execution_circuit = fuse_circuit(circuit) if self._fusion else circuit
        for operation in execution_circuit:
            state.apply(operation)
        return state


__all__ = ["StateVectorSimulator"]

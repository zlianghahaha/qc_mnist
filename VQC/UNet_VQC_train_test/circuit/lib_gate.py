import numpy as np
import qiskit



class ExtendGate():
    
    ################ Weiwen on 06-02-2021 ################
    # Function: ccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement ccz gate, which will flip the
    #       sign of state |111>
    # Parameters: (1) quantum circuit; 
    #             (2-3) control qubits;
    #             (4) target qubits;
    #             (5) auxiliary qubits.
    ######################################################
    @classmethod
    def ccz(cls,circ, q1, q2, q3, aux1):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.cz(aux1, q3)
        # cleaning the aux bit
        circ.ccx(q1, q2, aux1)
        return circ

    @classmethod
    def cccx(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cx(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement cccz gate, which will flip the
    #       sign of state |1111>
    # Parameters: (1) quantum circuit; 
    #             (2-4) control qubits;
    #             (5) target qubits;
    #             (6-7) auxiliary qubits.
    ######################################################
    @classmethod
    def cccz(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 4 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cz(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 4
    # Note: using the basic Toffoli gate to implement ccccx
    #       gate. It is used to switch the quantum states
    #       of |11110> and |11111>.
    # Parameters: (1) quantum circuit; 
    #             (2-5) control qubits;
    #             (6) target qubits;
    #             (7-8) auxiliary qubits.
    ######################################################
    @classmethod
    def ccccx(cls,circ, q1, q2, q3, q4, q5, aux1, aux2):
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, q4, aux2)
        circ.ccx(aux2, aux1, q5)
        # cleaning the aux bits
        circ.ccx(q3, q4, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: neg_weight_gate from Listing 3
    # Note: adding NOT(X) gate before the qubits associated
    #       with 0 state. For example, if we want to flip 
    #       the sign of |1101>, we add X gate for q2 before
    #       the cccz gate, as follows.
    #       --q3-----|---
    #       --q2----X|X--
    #       --q1-----|---
    #       --q0-----z---
    # Parameters: (1) quantum circuit; 
    #             (2) all qubits, say q0-q3;
    #             (3) the auxiliary qubits used for cccz
    #             (4) states, say 1101
    ######################################################
    @classmethod
    def neg_weight_gate(cls,circ,qubits,aux,state):
        idx = 0
        # The index of qubits are reversed in terms of states.
        # As shown in the above example: we put X at q2 not the third position.
        print("state 1",state)
        state = state[::-1]
        print("state 2",state)
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])
        cls.cccz(circ,qubits[0],qubits[1],qubits[2],qubits[3],aux[0],aux[1])
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])
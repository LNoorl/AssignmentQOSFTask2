# -*- coding: utf-8 -*-
import numpy as np
import copy
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error

"""
Assignment task 2 for the QOSF Mentorship Program

In this script we will construct a quantum circuit that has outputs 01 and 10 with each a probability of 0.5.
From literatur we already know that such a circuit can be constructed as follows, using an H-, an X- and a CNOT-gate:
"""

correct_circuit  = QuantumCircuit(2, 2)
correct_circuit.x(1)
correct_circuit.h(0)
correct_circuit.cx(0,1)
print(correct_circuit.draw())

""" 
Using a qasm simulator, we can experimentally verify that this circuit has the right outcomes: 
"""
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

device_backend = FakeVigo()

# The device coupling map is needed for transpiling to correct
# CNOT gates before simulation
coupling_map = device_backend.configuration().coupling_map

# Map the quantum measurement to the classical bits
correct_circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(correct_circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()
print(result.get_counts(correct_circuit))

"""
Indeed, when run multiple times, the counts of both '01' and '10' will lie around 500, 
while those of '00' and '11' will be zero.

However, the assignment was to find a circuit that only uses RX-, RY- and CNOT-gates. 
In the remaining of this script, we will use gradient descent to find the right parameters for the following ansatz circuit:
"""

theta0 = Parameter('θ0')
theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
theta3 = Parameter('θ3')

ansatz_circuit = QuantumCircuit(2, 2)
ansatz_circuit.rx(theta0, 0)
ansatz_circuit.rx(theta1, 1)   
ansatz_circuit.ry(theta2, 0)
ansatz_circuit.ry(theta3, 1)
ansatz_circuit.cx(0,1)
print(ansatz_circuit.draw())

"""
In this method, we use the qasm-simulator from qiskit to execute a given circuit (with noise when uncommented). 
We measure the values of '00', '01', '10' and '11'.
The qasm-simulator returns the results as strings.
We parse them as floats and return them in an array.
"""
def getIntegerCounts(circuit):
    
    # Example error probabilities
    p_reset = 0.03
    p_meas = 0.1
    p_gate1 = 0.05
    
    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)
    
    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["rx", "ry"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
  
    # Map the quantum measurement to the classical bits
    circuit.measure([0,1], [0,1])
    
    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, 
                  # noise_model=noise_bit_flip,
                       # basis_gates=noise_bit_flip.basis_gates,
                       shots=100)

    # Grab results from the job
    result = job.result()
    
    # Returns counts
    counts = result.get_counts(circuit)
    
    if "00" in counts:
         value00 = float(counts["00"])  
    else: 
        value00 = 0.0
        
    if "01" in counts:
         value01 = float(counts["01"])  
    else: 
        value01 = 0.0
        
    if "10" in counts:
         value10 = float(counts["10"])  
    else: 
        value10 = 0.0
        
    if "11" in counts:
         value11 = float(counts["11"])  
    else: 
        value11 = 0.0

    return [value00, value01, value10, value11]

"""
Performs N circuit simulations and returns the average outcome.
"""
def averageIntegerCounts(circuit,N):
    averageCounts = [0.0,0.0,0.0,0.0]
    for x in range(N):
        counts = getIntegerCounts(circuit)
        for y in range(4):
            averageCounts[y] = averageCounts[y]+counts[y]
    return [averageCounts[0]/N,averageCounts[1]/N,averageCounts[2]/N, averageCounts[3]/N]

"""
Defines the function needed to minimize for this problem. We want to obtain a circuit that has the following
averageIntegerCounts output: [0,50,50,0], hence we will minimize the sum
 averageIntegerCounts[0] + |averageIntegerCounts[1] - 50| +|averageIntegerCounts[2] - 50| + averageIntegerCounts[3]
"""
def circuitFunctionToMinimize(p0,p1,p2,p3,averagingOver=10):
    circuit = QuantumCircuit(2, 2)
    circuit.rx(p1, 0)
    circuit.rx(p2, 1)   
    circuit.ry(p3, 0)
    circuit.ry(p3, 1)
    circuit.cx(0,1)
    counts = averageIntegerCounts(circuit, averagingOver)
    return counts[0] + abs(counts[1]-50.0) + abs(counts[2]-50.0) + counts[3]

"""
Returns vector with delta added to j-th component vector[j].
"""
def plusDeltaAtJ(vector,j,delta):
    plusDelta = copy.copy(vector)
    jth_value = copy.copy(plusDelta[j])
    new_jth_value = jth_value+delta
    plusDelta[j] = new_jth_value
    return plusDelta

"""
Returns evaluation of a 4-variable function F at vectorA.
"""
def evaluateFAtVector(F,vectorA):
    return F(vectorA[0],vectorA[1],vectorA[2],vectorA[3])

"""
Computes the gradient of multivariable function F at vectorA using delta for approximating partial derivatives. 
"""
def computeGradient(F,vectorA,delta=0.01):
    
    # Number of variables
    N = len(vectorA)
     
    # Initialize gradient vector
    gradient = np.array([0.0] * N)
    
    for j in range(N):
        deltaVectorA = plusDeltaAtJ(vectorA,j,delta)
        gradient[j] = abs(evaluateFAtVector(F,deltaVectorA)-evaluateFAtVector(F, vectorA))/abs(delta)
    return gradient 

"""
Performs gradient descent algorithm using step size that starts with gamma, and divides gamma by 2 to make sure 
that the resulting sequence F(vectorA) >= F(newVectorA) >= ... is indeed not increasing.
"""            
def findMinimumUsingGradientDescent(F,initialParams,gamma = 1):
    vectorA = np.array(initialParams)
    valueAtA = copy.copy(evaluateFAtVector(F, vectorA))
    while valueAtA > 0.1:
        startingGamma = copy.copy(gamma)
        gradientAtA = computeGradient(F, vectorA,0.1);
        newVectorA = np.subtract(vectorA, (gamma*gradientAtA))
        while valueAtA <= evaluateFAtVector(F,newVectorA):
            gamma = gamma/2
            newVectorA = np.subtract(vectorA, (gamma*gradientAtA))
        vectorA = copy.copy(newVectorA)
        valueAtA = copy.copy(evaluateFAtVector(F, vectorA))
    return vectorA

# Starting parameters all zero, 1 measurement
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,1),[0.0,0.0,0.0,0.0]))

# Starting parameters all zero, 10 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,10),[0.0,0.0,0.0,0.0]))

# Starting parameters all zero, 100 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,100),[0.0,0.0,0.0,0.0]))

# Starting parameters all zero, 1000 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,1000),[0.0,0.0,0.0,0.0]))

random_params = np.random.normal(0., 2*np.pi, 4)

# Starting parameters random, 1 measurement
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,1),random_params))

# Starting parameters random, 10 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,10),random_params))

# Starting parameters random, 100 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,100),random_params))

# Starting parameters random, 1000 measurements
print(findMinimumUsingGradientDescent(lambda t0,t1,t2,t3: circuitFunctionToMinimize(t0,t1,t2,t3,1000),random_params))

"""
Discussion

Unfortunately, my results are not yet converging to answers, even without
explicitly adding noise to the simulations. I have tried tweaking the various parameters (delta, gamma),
but this did not help much. I would expect that a higher number of measurements per iteration would make 
the gradient descent algorithm converge faster. Also, I would expect that noise in the measurements would be compenstated
faster when we use a higher humber of measurements per iteration. 

Bonus question

In order to make sure that we produce state  |01⟩  +  |10⟩  and not any other combination of |01> + e(i*phi)|10⟩, 
we need to impose an extra condition on the parameters. Namely, the various phases they produce should add up so that
the qubits have a relative phase difference of zero. This extra condition can be included in the function that
we minimize using gradient descent. 

"""

    


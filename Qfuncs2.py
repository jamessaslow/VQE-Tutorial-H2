# Importing Packages
from qiskit import IBMQ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, transpiler
import math as m
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
import csv
from tqdm import tqdm
from mycolorpy import colorlist as mcp
import pandas as pd

S_simulator = Aer.backends(name='statevector_simulator')[0]
# Import all packages from QC Test



def ReturnPsi(obj, **kwargs):
    '''
    Returns the wavefunction psi in a numpy array
    '''

    if(type(obj) == QuantumCircuit  ):
        statevec = execute( obj, S_simulator, shots=1 ).result().get_statevector()
    if(type(obj) == np.ndarray):
        statevec = obj
    #==============
    statevec = list( np.asarray(statevec) )
    #==============
    dec = 5
    if 'precision' in kwargs:
        dec = int( kwargs['precision'] )
           
    # Rounding off states
    statevec = np.round(statevec,dec) 
    num_bases = len(statevec)
    num_qubits = int(np.log2(num_bases))
    
    
    def bin_gen(number, num_qubits):
        bin1 = bin(number)[2:]
        L = num_qubits - len(bin1)
        bin2 = L*'0' + bin1
        return bin2
    
    def rev_bin_gen(number, num_qubits):
        return bin_gen(number, num_qubits)[::-1]
    
    def binto10(number):
        num_qubits = len(number)
        num_bases  = int(2**num_qubits)
        reg_order = []
        for i in range(num_bases):
            reg_order.append(bin_gen(i, num_qubits))
            
        reg_order = np.array(reg_order)
        return np.where(reg_order == number)[0][0]
    
    def dual_sort(key, values):
        k=1
        while k!=0:
            k=0
            for i in range(len(key)-1):
                a = key[i]
                b = key[i+1]

                c = values[i]
                d = values[i+1]

                if b<a:
                    k+=1
                    key[i] = b
                    key[i+1] = a

                    values[i] = d
                    values[i+1] = c
                    
                    
        return values
    
    key = []
    for i in range(num_bases):
        key.append(binto10(rev_bin_gen(i, num_qubits)))
        
    psi = dual_sort(key, statevec)
    
    return psi
    





def FilterDuplicates(qc, **kwargs):
    '''
    Description:
    Used as an intermediate functions within other functions in this package
    The purpose of FilterDuplicates is to return a compressed version of psi that contains unique states only.
    The density of each state as well as the filtered braket state name is also returned
    
    kwargs -> return specific items if listed
    
    Use: Intermediate function
    '''
    psi = ReturnPsi(qc)
    num_bases = len(psi) # Number of basis states
    
    states = []
    for i in range(num_bases):
        states.append(base10to2(i,int(np.log2(num_bases)) ))
    
    delete_array = [] # List of Indices of duplicates to be deleted from psi
    
    # Performing boolean logic to remove duplicates and perserving binary ordering
    for i in range(len(psi)):
        for j in range(len(psi)):
            if i == j:
                continue
            if psi[i] == psi[j]:
                if i not in delete_array and j not in delete_array:
                    delete_array.append(j)
            

    psi_filtered = np.delete(psi, delete_array) # Filtering out duplicates
    N_filtered   = len(psi_filtered)            # Length of unique basis states
    states_filtered = np.delete(states, delete_array)

    density = [] # Recording the number of duplicates at a particular filtered state
    for i in range(N_filtered):
        density.append(len(np.where(psi_filtered[i] == psi)[0]))
    
    return psi_filtered, density, states_filtered
    
    
    
    
    
    
    
def PlotPsi(qc, **kwargs):
    '''
    Description:
    Does a complex amplitude plot of psi
    
    **kwargs:
    density: If set to true, will color code each unique state based on the repeatibility of the same state
    If set to false, will uniquely color code each unique state to be self consistent with the color code in PrintPsi
    '''
    
    # Presetting logical statements to be changed by **kwargs
    density_bool = kwargs.get('density', False)
    
    
    psi = ReturnPsi(qc)
    num_bases = len(psi) # Number of basis vectors
    
    psi_filtered, density, states_filtered = FilterDuplicates(qc) # Gathering unique points & Density of each pointS
    # Use density 
    N_filtered = len(psi_filtered)
    
    
    # Selecting Real and Imaginary Components of filtered psi
    x = np.real(psi_filtered)
    y = np.imag(psi_filtered)
    
    
    # Constructing a reference circle
    r_circ = 1/np.sqrt(num_bases)
    theta_circ = np.linspace(0,2*np.pi,100)
    x_circ = r_circ*np.cos(theta_circ)
    y_circ = r_circ*np.sin(theta_circ)
        
        
    # Prompting matplotlib figure
    fig, ax = plt.subplots(figsize = (6,6))    
    
    
    # Color matching unique states
    if density_bool == False:     
        colormap = plt.get_cmap('brg')
        colors = colormap(np.linspace(0, 1, N_filtered))
    
    # Color matching the density of states
    if density_bool == True:
        colors = density
    
    
    plt.title('Amplitude Plot')
    plt.plot(x_circ, y_circ, color = 'black', alpha = 0.3) # Plotting a Unit Circle
    plt.scatter(x,y, c=colors) # Scattering Points for Each State
    
    if density_bool == True:
        cbar = plt.colorbar() # Shows overlap of identical bases in density colorbar
        cbar.ax.set_ylabel('Density', rotation=90)
    
    plt.plot([0],[0], marker = '+', linestyle = '', color = 'red', markersize = 20)
    plt.xlabel('$Re$')    
    plt.ylabel('$Im$')
    plt.grid()
    plt.show()
    
    
    

    
    


# Making a function that converts base 10 to base 2 and uses braket notation
def base10to2(number,num_qubits):
    bin1 = bin(number)[2:]
    L = num_qubits - len(bin1)
    bin2 = L*'0' + bin1
    return '|' + bin2 + '>'







def PrintPsi(qc, **kwargs):
    '''
    Description:
    Returns a pandas dataframe of information regarding the wavefunction with a colorcode that is self consistent with
    the PlotPsi function
    
    **kwargs:
    If unique = True, pandas dataframe will only display unique states
    If unique = False, pandas dataframe will display every state
    
    If extract = True, no color modifications will be displayed, but data can be extracted
    If extract = False, color modifications will be displayed, but user can't extract data
    
    '''
    
    # Presetting logical statements to be changed by **kwargs
    unique_bool = kwargs.get('unique', False)
    extract_bool = kwargs.get('extract', False)
    
    psi = ReturnPsi(qc)
    num_bases = len(psi) # Number of basis vectors
    num_qubits = int(np.log2(num_bases))


    psi_filtered, density, states_filtered = FilterDuplicates(qc) # Gathering unique points & Density of each pointS
    N_filtered = len(psi_filtered)    

    colormap = plt.get_cmap('brg')
    colors = colormap(np.linspace(0, 1, N_filtered))    

    # Generating name tags for each basis vector
    states = []
    for i in range(num_bases):
        states.append(base10to2(i,num_qubits))

        
    # If we want to display unique states   
    if unique_bool == True:
        states = states_filtered
        psi_real = np.real(psi_filtered)
        psi_imag = np.imag(psi_filtered)
        psi_mag  = abs(psi_filtered)
        theta    = np.angle(psi_filtered)
        probs    = psi_mag**2

        
    # If we want to display all states
    if unique_bool == False:
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        psi_mag  = abs(psi)
        theta    = np.angle(psi)
        probs    = psi_mag**2
    
    
    # Look into how to add additional columns on to already existing pandas dataframes (i.e. magnitude, angle, density, etc)
    # Constructing a Pandas DataFrame
    df = pd.DataFrame({'State': states, 'Real Part': psi_real, 'Imag Part': psi_imag,
                      'Magnitude': psi_mag, 'Phase': theta, 'Probability': probs})
        
    
    # Converting from RGB to Hex
    colors_hex = []
    for i in range(len(colors)):
        colors_hex.append( str(mcolors.rgb2hex(colors[i])) )

    # Making a dictionary between filtered states and hex colors
    state_color_map = dict(zip(states_filtered, colors_hex))


    # A Function that highlights each quantum state
    def color_state(row):
        state = row['State']
        color = state_color_map.get(state, 'unknown')
        return ['background-color: %s' % color] + [''] * (len(row) - 1)   # Highlights each state cell

    if extract_bool == False:
        styled_df = df.style.apply(color_state, axis=1)
        return styled_df
    else:
        return df
    
    
    
    
def HistPsi(qc):
    '''
    Description:
    Plots a histogram of probabilities associated with the wavefunction
    '''
    
    psi = ReturnPsi(qc)
    num_bases = len(psi)
    num_qubits = int(np.log2(num_bases))
    probs = abs(psi)**2
    
    states = []
    for i in range(num_bases):
        states.append( base10to2(i, num_qubits) )
    
    plt.bar(states, probs)
    plt.xticks(rotation = 45)
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.show
    
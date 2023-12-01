import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import xlsxwriter
import os.path
from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



choice = int(input("Do you want to run default value or want to give manual value? Press 1 for default and 2 for manual entry: "))
if choice == 2:
    path0 = 'C:\\Users\\Acer\\AppData\\Local\\Programs\\Python\\Python311\\'
    f_name = (input("Enter the file name:"))
    f_path = path0+f_name

    workbook = pd.read_excel(os.path.basename(f_path))
    workbook.head()
    
    e_total = int(input("Enter the total number of elements: "))
    l_total0 = int(input("Enter the total length: "))
    mesh_number = int(input("Enter mesh number: "))

    e_name = np.empty(e_total, dtype='object')
    e_length = np.zeros(e_total)

    # Taking each element name and corresponding length
    for m in range(0, e_total):
        print("For element number", m+1)
        e_name[m] = input("Enter the element name: ")
        e_length[m] = int((float(input("Enter the element length: "))*mesh_number)/l_total0)
    e_length_t = np.zeros(e_total)
    
    sum = 0
    for m in range(0, e_total):
        e_length_t[m] = sum + e_length[m]
        sum = e_length_t[m]
        
    l_total = int(np.sum(e_length))

    # Creating the image of user defined gemoetry
    img = Image.new("RGB", (int(l_total),int(l_total/4)), "white")
    lay = np.empty(e_total, dtype="object")
    for m in range(0, e_total):
        if e_name[m].find('R') != -1 or e_name[m].find('CR') != -1 or e_name[m].find('C') != -1:
            if e_name[m].find('C') != -1:
                color = 'white'
            if e_name[m].find('R') != -1:
                color = 'black'
            if e_name[m].find('CR') != -1:
                color = 'gray'    
        elif e_name[m].find('F') != -1:
            if e_name[m].find('F1') != -1:
                color = 'lightskyblue'
            elif e_name[m].find('F2') != -1:
                color = 'greenyellow'
            elif e_name[m].find('F3') != -1:
                color = 'orange'
            else:
                color = 'yellow'
            
        lay[m] = Image.new("RGB", (int(e_length[m]), int(l_total/4)), color)
        if m == 0:
            img.paste(lay[m], (0,0))
        else:    
            img.paste(lay[m], (int(e_length_t[m-1]),0))
    img.show()

    Coefficient1 = np.zeros([l_total,l_total])
    Coefficient2 = np.zeros([l_total,l_total])

    Phi1_0 = np.ones([l_total,1])
    Phi1_1 = np.ones([l_total,1])
    Phi2_0 = np.ones([l_total,1])
    Phi2_1 = np.ones([l_total,1])


    v1F1 = np.zeros([l_total,l_total])
    v2F2 = np.zeros([l_total,l_total])


    Scattering_CS21 = np.zeros([l_total,l_total])
    Scattering_CS12 = np.zeros([l_total,l_total])

    k_0 = 1
    k_1 = 0
    k_Matrix = np.ones([24000,1])

    Convergence = 1

    w = 1.8
else:
    print('Calculation will be performed by default value')
    mesh_number = 420

    Coefficient1 = np.zeros([420,420])
    Coefficient2 = np.zeros([420,420])

    Phi1_0 = np.ones([420,1])
    Phi1_1 = np.ones([420,1])
    Phi2_0 = np.ones([420,1])
    Phi2_1 = np.ones([420,1])


    v1F1 = np.zeros([420,420])
    v2F2 = np.zeros([420,420])


    Scattering_CS21 = np.zeros([420,420])
    Scattering_CS12 = np.zeros([420,420])

    k_0 = 1
    k_1 = 0
    k_Matrix = np.ones([24000,1])

    Convergence = 1

    w = 1.8


# Creating excel output file
def toExcel(matrix, name):
    file = pd.DataFrame(matrix)
    filepath = name+'.xlsx'
    file.to_excel(filepath, index = False)


# For inputing scattering cross-section
def S_CS(g, i):    
    y = np.ones(1)
    x = 0
    if choice == 2:
        if g==1:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[2]
                    x = y[0]
                    break
        if g==2:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[3]
                    x = y[0]
                    break

        
    else:
        if g==1:    # For Fast neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0.0255380
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0181930
            elif i >= 401 and i <= 420:    # For reflector  
                x = 0.0255220   
            else:
                x = 0

        if g==2:    # For Thermal neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0.0001245
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0013089
            elif i >= 401 and i <= 420:    # For reflector
                x = 0.0001231
            else:
                x = 0
    
    return x


# For inputing diffusion coefficient
def D_CF(g, i):    
    y = np.ones(1)
    x = 0
    if choice == 2:
        if g==1:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[0]
                    x = y[0]
                    break
        if g==2:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[1]
                    x = y[0]
                    break
    else:
        if g==1:    # For Fast neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 1.1245305
            elif i >= 21 and i <= 400:    # For Fuel
                x = 1.0933622
            elif i >= 401 and i <= 420:    # For reflector  
                x = 1.1251378   
            else:
                x = 0

        if g==2:    # For Thermal neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0.7503114
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.3266693
            elif i >= 401 and i <= 420:    # For reflector
                x = 0.7501763
            else:
                x = 0

    return x


# For inputing absorption cross-section
def Abs_CS(g, i):    
    y = np.ones(1)
    x = 0
    if choice == 2:
        if g==1:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[4]
                    x = y[0]
                    break
        if g==2:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[5]
                    x = y[0]
                    break
    else:
        if g==1:    # For Fast neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0.0008996
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0092144
            elif i >= 401 and i <= 420:    # For reflector  
                x = 0.0008984   
            else:
                x = 0

        if g==2:    # For Thermal neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0.0255590
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0778104
            elif i >= 401 and i <= 420:    # For reflector
                x = 0.0255600
            else:
                x = 0

    return x


# For inputing neutron density times fission cross-section
def vF(g, i):    
    y = np.ones(1)
    x = 0
    if choice == 2:
        if g==1:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[6]
                    x = y[0]
                    break
        if g==2:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[7]
                    x = y[0]
                    break
    else:                
        if g==1:    # For Fast neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0065697
            elif i >= 401 and i <= 420:    # For reflector  
                x = 0  
            else:
                x = 0

        if g==2:    # For Thermal neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.1312600
            elif i >= 401 and i <= 420:    # For reflector
                x = 0
            else:
                x = 0
    return x


# For inputing fission cross-section
def Fission_CS(g, i):    
    y =np.ones(1)
    x = 0
    if choice == 2:
        if g==1:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[8]
                    x = y[0]
                    break
        if g==2:
            for m in range(0, e_total):
                if i >= 1 and i <= e_length_t[m]:
                    y[0] = workbook[{e_name[m]}].iloc[9]
                    x = y[0]
                    break
    else:
        if g==1:    # For Fast neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0025763
            elif i >= 401 and i <= 420:    # For reflector  
                x = 0   
            else:
                x = 0

        if g==2:    # For Thermal neutron
            if i >= 1 and i <= 20:    # For reflector
                x = 0
            elif i >= 21 and i <= 400:    # For Fuel
                x = 0.0538660
            elif i >= 401 and i <= 420:    # For reflector
                x = 0
            else:
                x = 0

    return x


#For inputting height
def h(i):   
    x = 1
    return x



def Coefficient1Matrix():
    m = 0
    n = 0
    g = 1

    if choice == 2:
        for m in range (0,l_total):
            i = m+1
            for n in range (0,l_total):
                if n == m-1:
                    Coefficient1[m][n] = -(2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i))
                                                 
                elif n == m:
                    Coefficient1[m][n] = (2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i)) + (2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i)) + h(i)*(Abs_CS(g,i)+S_CS(g,i))
                            
                elif n == m+1:
                    Coefficient1[m][n] = -(2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i))

                else:
                    Coefficient1[m][n] = 0
    else:
        for m in range (0,420):
            i = m+1
            for n in range (0,420):
                if n == m-1:
                    Coefficient1[m][n] = -(2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i))
                                                     
                elif n == m:
                    Coefficient1[m][n] = (2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i)) + (2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i)) + h(i)*(Abs_CS(g,i)+S_CS(g,i))
                                
                elif n == m+1:
                    Coefficient1[m][n] = -(2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i))

                else:
                    Coefficient1[m][n] = 0
    




def Coefficient2Matrix():
    m = 0
    n = 0
    g = 2

    if choice == 2:
        for m in range (0,l_total):
            i = m+1
            for n in range (0,l_total):
                if n == m-1:
                    Coefficient2[m][n] = -(2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i))
                                                 
                elif n == m:
                    Coefficient2[m][n] = (2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i)) + (2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i)) + h(i)*(Abs_CS(g,i)+S_CS(g,i))
                            
                elif n == m+1:
                    Coefficient2[m][n] = -(2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i))

                else:
                    Coefficient2[m][n] = 0
    else:
        for m in range (0,420):
            i = m+1
            for n in range (0,420):
                if n == m-1:
                    Coefficient2[m][n] = -(2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i))
                                                     
                elif n == m:
                    Coefficient2[m][n] = (2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i)) + (2*D_CF(g,i)*D_CF(g,i-1))/(D_CF(g,i)*h(i-1)+D_CF(g,i-1)*h(i)) + h(i)*(Abs_CS(g,i)+S_CS(g,i))
                                
                elif n == m+1:
                    Coefficient2[m][n] = -(2*D_CF(g,i)*D_CF(g,i+1))/(D_CF(g,i)*h(i+1)+D_CF(g,i+1)*h(i))

                else:
                    Coefficient2[m][n] = 0
    



#From here, starting work on the right side of the equation

def v1F1Matrix():
    m = 0
    n = 0
    g = 1

    if choice == 2:
        for m in range (0, l_total):
            i = m+1
            for n in range (0, l_total):
                if n == m:
                    v1F1[m][n] = vF(g,i)
                else:
                    v1F1[m][n] = 0
    else:
        for m in range (0, 420):
            i = m+1
            for n in range (0, 420):
                if n == m:
                    v1F1[m][n] = vF(g,i)
                else:
                    v1F1[m][n] = 0
    return v1F1
    


def v2F2Matrix():
    m = 0
    n = 0
    g = 2

    if choice == 2:
        for m in range (0,l_total):
            i = m+1
            for n in range (0,l_total):
                if n == m:
                    v2F2[m][n] = vF(g,i)
                else:
                    v2F2[m][n] = 0
    else:
        for m in range (0,420):
            i = m+1
            for n in range (0,420):
                if n == m:
                    v2F2[m][n] = vF(g,i)
                else:
                    v2F2[m][n] = 0
    return v2F2



def Scattering_CS12Matrix():
    m = 0
    n = 0
    g = 1

    if choice == 2:
        for m in range (0,l_total):
            i = m+1
            for n in range (0,l_total):
                if n == m:
                    Scattering_CS12[m][n] = S_CS(g,i)
                else:
                    Scattering_CS12[m][n] = 0
    else:
        for m in range (0,420):
            i = m+1
            for n in range (0,420):
                if n == m:
                    Scattering_CS12[m][n] = S_CS(g,i)
                else:
                    Scattering_CS12[m][n] = 0
    return Scattering_CS12



def Scattering_CS21Matrix():
    m = 0
    n = 0
    g = 2

    if choice == 2:
        for m in range (0,l_total):
            i = m+1
            for n in range (0,l_total):
                if n == m:
                    Scattering_CS21[m][n] = S_CS(g,i)
                else:
                    Scattering_CS21[m][n] = 0
    else:
        for m in range (0,420):
            i = m+1
            for n in range (0,420):
                if n == m:
                    Scattering_CS21[m][n] = S_CS(g,i)
                else:
                    Scattering_CS21[m][n] = 0
    return Scattering_CS21


# Solving the value for Phi_1
def group_1_solver (k, Phi1, Phi2):
    x = np.matmul(v1F1, Phi1)
    y = np.matmul(v2F2, Phi2)
    z = np.matmul(Scattering_CS21, Phi2)

    
    RHS = x/k + y/k + z
    InvertedCoefficient1 = np.linalg.inv(Coefficient1)
    Phi1_1 = np.matmul(InvertedCoefficient1, RHS)  
    Phi1_1 = w*np.matmul(InvertedCoefficient1, RHS) + (1-w)*Phi1_1

    return Phi1_1


# Solving the value for Phi_2
def group_2_solver (Phi1):

    RHS = np.matmul(Scattering_CS12, Phi1)
    InvertedCoefficient2 = np.linalg.inv(Coefficient2)
    Phi2_1 = np.matmul(InvertedCoefficient2, RHS)  
    Phi2_1 = w*np.matmul(InvertedCoefficient2, RHS) + (1-w)*Phi2_1

    return Phi2_1


# Calculating multiplication factor
def k_calculator(k0, P10, P20, P11, P21):
    x_1 = np.matmul(v1F1, P11)
    y_1 = np.matmul(v2F2, P21)
    x_0 = np.matmul(v1F1, P10)
    y_0 = np.matmul(v2F2, P20)

    z_1 = np.sum(x_1) + np.sum(y_1)
    z_0 = np.sum(x_0) + np.sum(y_0)


    k1 = k0*z_1/z_0

    return k1


def check_convergence(k0, k1):
    E = (k1 - k0) / k1
    return E




def main():

    start = time.time()

    Coefficient1Matrix()
    Coefficient2Matrix()
    v1F1Matrix()
    v2F2Matrix()
    Scattering_CS12Matrix()
    Scattering_CS21Matrix()

    global k_0, Phi1_0, Phi2_0

    Phi1_1 = group_1_solver(k_0, Phi1_0, Phi2_0)
    Phi2_1 = group_2_solver(Phi1_1)
    k_1 = k_calculator(k_0, Phi1_0, Phi2_0, Phi1_1, Phi2_1)

    

    Convergence = check_convergence(k_0, k_1)
    
    if choice == 2:
        Desired_Convergence = math.pow(10, -7)
    else:
        Desired_Convergence = math.pow(10, -7)

    Count = 0

    inner_count1 = 0
    inner_count2 = 0

    k_plot = np.empty(24000, dtype='object')

    while (abs(Convergence) > Desired_Convergence) or (inner_count1<mesh_number/2 if choice==2 else inner_count1<420) or (inner_count2<mesh_number/2 if choice==2 else inner_count2<420):
        k_0 = k_1
        Phi1_0 = Phi1_1
        Phi2_0 = Phi2_1
        inner_count1 = 0
        inner_count2 = 0

        Phi1_1 = group_1_solver(k_0, Phi1_0, Phi2_0)
        if choice == 2:
            for m in range(0, l_total):
                for n in range(0, 1):
                    if (abs(Phi1_1[m][n] - Phi1_0[m][n]))/Phi1_1[m][n] < math.pow(10, -3):
                        inner_count1 +=1
            Phi2_1 = group_2_solver(Phi1_1)
            for m in range(0, l_total):
                for n in range(0, 1):
                    if (abs(Phi2_1[m][n] - Phi2_0[m][n]))/Phi2_1[m][n] < math.pow(10, -3):
                        inner_count2 +=1
        else:
            for m in range(0, 420):
                for n in range(0, 1):
                    if (abs(Phi1_1[m][n] - Phi1_0[m][n]))/Phi1_1[m][n] < math.pow(10, -7):
                        inner_count1 +=1
            Phi2_1 = group_2_solver(Phi1_1)
            for m in range(0, 420):
                for n in range(0, 1):
                    if (abs(Phi2_1[m][n] - Phi2_0[m][n]))/Phi2_1[m][n] < math.pow(10, -7):
                        inner_count2 +=1
        k_1 = k_calculator(k_0, Phi1_0, Phi2_0, Phi1_1, Phi2_1)

        Convergence = check_convergence(k_0, k_1)

        k_Matrix[Count][0] = k_1
        k_plot[Count] = k_1

        Count += 1

    toExcel(Phi1_1, "Fast")
    toExcel(Phi2_1, "Thermal")
    toExcel(k_Matrix, "Multiplication_Factor")
    toExcel(Phi1_0, "Fast0")
    toExcel(Phi2_0, "Thermal0")


    if choice == 2:
        x = np.ones([l_total,1])
        for m in range(0,l_total):
            for n in range(0, 1):
                x[m][n] = m
        y = np.ones([24000,1])
        for m in range(0,Count):
            for n in range(0, 1):
                y[m][n] = m
    else:
        x = np.ones([420,1])
        for m in range(0,420):
            for n in range(0, 1):
                x[m][n] = m
        y = np.ones([24000,1])
        for m in range(0,Count):
            for n in range(0, 1):
                y[m][n] = m

    a = np.matmul(v1F1, Phi1_1)
    b = np.matmul(v2F2, Phi2_1)
    p = (a+b)*200
    

    print("Multiplication factor is", k_1)
    print("Convergence Error", Convergence)
    print("Number of iteration is", Count)
    print("Number of inner convergence for Phi_1", inner_count1)
    print("Number of inner convergence for Phi_2", inner_count2)

    end = time.time()
    print("Execution time is", end-start)

    #plot 1:
    plt.subplot(2, 2, 1)
    plt.plot(x, Phi1_1)
    plt.xlabel("Mesh Number")
    plt.ylabel("Flux Profile")
    plt.title("Fast Neutron Flux")
    

    #plot 2:
    plt.subplot(2, 2, 2)
    plt.plot(x, Phi2_1)
    plt.xlabel("Mesh Number")
    plt.ylabel("Flux Profile")
    plt.title("Thermal Neutron Flux")
    

    #plot 3:
    plt.subplot(2, 2, 3)
    plt.plot(y, k_plot)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Multiplication Factor")
    plt.title("Multiplication Factor Convergence")
    

    #plot 4:
    plt.subplot(2, 2, 4)
    plt.plot(x, p)
    plt.xlabel("Mesh Number")
    plt.ylabel("Power Profile")
    plt.title("Power Density Distribution")
    

    plt.subplots_adjust(hspace=0.388)
        

    #plot 5:
    plt.figure("Flux")
    plt.plot(x, Phi1_1, color='m', label='fast')
    plt.plot(x, Phi2_1, color='g', label='thermal')
    plt.legend()
    plt.xlabel("Mesh Number")
    plt.ylabel("Flux Profile")
    plt.title("Fast and Thermal Neutron Flux Profile")
    plt.show()
    

main()

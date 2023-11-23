
from mpi4py import MPI
from lammps import lammps
from lammps import PyLammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

import matplotlib.pyplot as plt

import numpy as np
from scipy import integrate

import sys

def sum_over_MPI(A, irank, root=0):
    #Put data into contiguous c arrays read to send through MPI
    sendbuf = np.ascontiguousarray(A)
    recvbuf = np.ascontiguousarray(A)
    comm.Reduce([sendbuf, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], op=MPI.SUM, root=root )
    #Summed arrays only exist on root process, unpack into variables
    if irank == root:
        return recvbuf
    else:
        return None

# Should this be float(Count) in denominator?
def update_var(partial, mean, var, Count):
    return (Count-1)/float(Count)*var + ((Count-1)*((partial - mean)/float(Count))**2)

def update_mean(partial, mean, Count):
    return ((Count-1)*mean + partial)/float(Count)


Nsteps_Child=1000
Delay=10
Nsteps_eff=int(Nsteps_Child/Delay)+1
Nbins=100
Nchunks=100
Nchildren=10
Nmappings=4
avetime_ncol = 2
avechunk_ncol = 3
avechunks_nrows=1
stepsize_child = 1
shear_rate = 1
dt = 0.0025

maps=[0,7,36,35]

#This code is run using MPI - each processes will
#run this same bit code with its own memory
comm = MPI.COMM_WORLD
irank = comm.Get_rank()
nprocs = comm.Get_size()
root = 0
print("Proc {:d} out of {:d} procs".format(irank+1,nprocs))




DAV_response_mean  = np.zeros([Nsteps_eff, avetime_ncol])
DAV_profile_mean   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_response_mean  = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_mean   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

DAV_response_var  = np.zeros([Nsteps_eff, avetime_ncol])
DAV_profile_var   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

TTCF_response_var  = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_var   = np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

data_response = np.zeros([Nmappings, Nsteps_eff, avetime_ncol])
data_profile  = np.zeros([Nmappings, Nsteps_eff, Nbins, avechunk_ncol])
TTCF_response_partial = np.zeros([Nsteps_eff, avetime_ncol])
TTCF_profile_partial= np.zeros([Nsteps_eff, Nbins, avechunk_ncol])

np.random.seed(irank)
seed = str(int(np.random.randint(1, 1e5 + 1)))

args = ['-sc', 'none','-log', 'none','-var', 'perturbation_seed' , seed , 'var' , 'srate', shear_rate]
lmp = lammps(comm=MPI.COMM_SELF, cmdargs=args)
L_script = PyLammps(ptr=lmp)
nlmp = lmp.numpy 

            
lmp.file("mother+daughter.in")
lmp.command("run " + str(1500))
            
lmp.command("unfix NVT_decorrelation")
lmp.command("fix snapshot all store/state 0 x y z vx vy vz")

Count = 0

for Nc in range(1,Nchildren+1,1):
               
    lmp.command("include ./load_state.lmp")
    lmp.command("fix NVT_sampling all nvt temp ${T} ${T} ${Thermo_damp} tchain 1")
    lmp.command("run " + str(1000))
    lmp.command("unfix NVT_sampling")
    lmp.command("fix snapshot all store/state 0 x y z vx vy vz")

    for Nm in range(Nmappings):
    
        TTCF_response_partial[0, :]=0
        TTCF_profile_partial[0, :, :]=0
    
        lmp.command("variable map equal " + str(maps[Nm]))
        lmp.command("variable child_index equal " + str(Nc))
        lmp.command("include ./load_state.lmp")
        lmp.command("include ./mappings.lmp")
        lmp.command("include ./set_daughter.lmp")
        

        #lmp.command("run " + str(0))
        #Get initial value
        for column in range(avetime_ncol):
            data_response[Nm, 0, column] = nlmp.extract_fix("Response", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)
          
        for y in range(Nbins):
            for column in range(avechunk_ncol):
                data_profile[Nm, 0, y, column] = nlmp.extract_fix("Profile", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, y , column)
                
        omega = data_response[Nm, 0, -1] 

        #Run over time        
        for t in range(1 , Nsteps_eff , 1):
            lmp.command("run " + str(Delay))
            for column in range(avetime_ncol):
                data_response[Nm, t, column] = nlmp.extract_fix("Response", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, column)
                
            for y in range(Nbins):
                for column in range(avechunk_ncol):
                    data_profile[Nm, t, y, column] = nlmp.extract_fix("Profile", LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, y , column)

            #INTEGRATION PROCESS: I INTEGRATE EACH SINGLE TRAJECTORY, SO THAT I HAVE A RELIABLE ESTIMATE OF THE VARIANCE
            if (t % 2) == 0:
                TTCF_profile_partial[t,:,:]  = ( TTCF_profile_partial[t-2,:,:] 
                                                + omega*Delay*dt/3*( data_profile[Nm,t-2,:,:] 
                                                                 + 4*data_profile[Nm,t-1,:,:] 
                                                                   + data_profile[Nm,t,:,:] ))
                TTCF_response_partial[t,:]   = (TTCF_response_partial[t-2,:]  
                                                + omega*Delay*dt/3*( data_response[Nm ,t-2 ,:] 
                                                                 + 4*data_response[Nm ,t-1 ,:] 
                                                                   + data_response[Nm ,t ,:]))
                TTCF_profile_partial[t-1,:,:]  = ( TTCF_profile_partial[t-2,:,:] + TTCF_profile_partial[t,:,:] )/2
                TTCF_response_partial[t-1,:]   = ( TTCF_response_partial[t-2,:]  + TTCF_response_partial[t,:] )/2
        
        lmp.command("include ./unset_daughter.lmp")

        partial_profile  = data_profile[Nm,:,:,:]
        partial_response = data_response[Nm,:,:]
        Count += 1

        #Should TTCF_profile_mean, etc be on right hand side as well as left?
        #You defined these as empty initially so undefined. Use zeros if it should.
        #TTCF_profile_var+=((Count-1)/Count)*((TTCF_profile_partial - TTCF_profile_mean)**2)        
        #TTCF_profile_mean+=((Count-1)/Count)*TTCF_profile_mean + TTCF_profile_partial/Count
        
        #TTCF_response_var+=((Count-1)/Count)*((TTCF_response_partial - TTCF_response_mean)**2)        
        #TTCF_response_mean+=((Count-1)/Count)*TTCF_response_mean + TTCF_response_partial/Count
        
        #DAV_profile_var+=((Count-1)/Count)*((partial_profile - DAV_profile_mean)**2)        
        #DAV_profile_mean+=((Count-1)/Count)*DAV_profile_mean + partial_profile/Count
        
        #DAV_response_var+=((Count-1)/Count)*((partial_response - DAV_response_mean)**2)        
        #DAV_response_mean+=((Count-1)/Count)*DAV_response_mean + partial_response/Count
 
        #Would may be simpler as a function
        TTCF_profile_var= update_var(TTCF_profile_partial, TTCF_profile_mean, TTCF_profile_var, Count)
        TTCF_profile_mean= update_mean(TTCF_profile_partial, TTCF_profile_mean, Count)
        
        DAV_profile_var= update_var(partial_profile, DAV_profile_mean, DAV_profile_var, Count)
        DAV_profile_mean= update_mean(partial_profile, DAV_profile_mean, Count)


        TTCF_response_var= update_var(TTCF_response_partial, TTCF_response_mean, TTCF_response_var, Count)
        TTCF_response_mean= update_mean(TTCF_response_partial, TTCF_response_mean, Count)
        
        DAV_response_var= update_var(partial_response, DAV_response_mean, DAV_response_var, Count)
        DAV_response_mean= update_mean(partial_response, DAV_response_mean, Count)
        
        #and you could define lists to make this even more concise (but less clear?)
        #var_list = [TTCF_profile_var, TTCF_response_var, DAV_profile_var, DAV_response_var]
        #mean_list = [TTCF_profile_mean, TTCF_response_mean, DAV_profile_mean, DAV_response_mean]
        #partials_list = [TTCF_profile_partial, TTCF_response_partial, partial_profile, partial_response]
        #for i in range(len(var_list)):
        #    var_list[i] += update_var(partials_list[i], mean_list[i], Count)
        #    mean_list[i] += update_var(partials_list[i], mean_list[i], Count)

lmp.close()
        
#I GET THE FINAL COLUMN BECAUSE BY DEFAULT LAMMPS GIVE YOU ALSO THE USELESS INFO ABOUT THE BINS. SINCE I HAVE ONLY ONE QUANTITY TO COMPUTE, I TAKE THE LAST.
# IF I HAD N QUANTITIES, I WOULD TAKE THE LAST N ELEMENTS

TTCF_profile_mean = TTCF_profile_mean[:,:,-1]
DAV_profile_mean  = DAV_profile_mean[:,:,-1]

TTCF_profile_var = TTCF_profile_var[:,:,-1]
DAV_profile_var  = DAV_profile_var[:,:,-1]

TTCF_response_var/= np.sqrt(Count)
DAV_response_var /= np.sqrt(Count)
TTCF_profile_var /= np.sqrt(Count)
DAV_profile_var  /= np.sqrt(Count)

### I HAVE COMPUTED MEN AND VARIANCE OF BOTH DAV AND TTCF WITHIN EACH SINGLE CORE, SO THAT IF YOU USE A SINGLE CORE YOU STILL HAVE AN ESTIMATE OF THE FLUCTUATIONS. 
### NOW WE NEED TO AVERAGE OVER THE DIFFERENT CORES. FOR THE MEAN, IT IS EASY, YOU SUM AND THE MEANS AND DIVIDE BY NCORES, FOR THE VARIANCE, YOU SUM THE VARIANCES AND DIVIDE BY THE SQRT(NCORES)
### ESSENTIALLY WE HAVE THE MEAN OF EACH CORE: M1,M2,M3,M4,... AND THE VARIANCE V1,V2,V3,V4,... 
### WHERE THE MEANS HAVE THE SUFFIX _mean AND THE VARIANCE _var, FOR EACH ARRAY, (TTCF_profile, TTCF_response, DAV_profile, DAV_response)
### SO THE TOTAL MEAN AND VARIANCE OVER THE CORES ARE TOT_MEAN=(M1+M2+M3+...)/NCORES AND TOT_VAR=(V1+V2+V3+...)/SQRT(NCORES)
TTCF_profile_mean_total = sum_over_MPI(TTCF_profile_mean, irank)
DAV_profile_mean_total = sum_over_MPI(DAV_profile_mean, irank)
TTCF_profile_var_total = sum_over_MPI(TTCF_profile_var, irank)
DAV_profile_var_total = sum_over_MPI(DAV_profile_var, irank)

TTCF_response_mean_total = sum_over_MPI(TTCF_response_mean, irank)
DAV_response_mean_total = sum_over_MPI(DAV_response_mean, irank)
TTCF_response_var_total = sum_over_MPI(TTCF_response_var, irank)
DAV_response_var_total = sum_over_MPI(DAV_response_var, irank)

#Total is None on everything but the root processor
if irank == root:
    TTCF_profile_mean_total = TTCF_profile_mean_total/float(nprocs) #Be careful of division by int in Python
    DAV_profile_mean_total  = DAV_profile_mean_total/float(nprocs)
    TTCF_profile_var_total  = TTCF_profile_var_total/np.sqrt(nprocs)
    DAV_profile_var_total   = DAV_profile_var_total/np.sqrt(nprocs)
    
    TTCF_response_mean_total = TTCF_response_mean_total/float(nprocs) #Be careful of division by int in Python
    DAV_response_mean_total  = DAV_response_mean_total/float(nprocs)
    TTCF_response_var_total  = TTCF_response_var_total/np.sqrt(nprocs)
    DAV_response_var_total   = DAV_response_var_total/np.sqrt(nprocs)
    

# This code animates the time history
#    plt.ion()
#    fig, ax = plt.subplots(1,1)
#    plt.show()
#    ft = True
#    for t in range(TTCF_profile_mean_total.shape[0]):
#        print(t)
#        l1, = ax.plot(DAV_profile_mean_total[t, :],'r-', label="DAV")
#        l2, = ax.plot(TTCF_profile_mean_total[t, :],'b-', label="TTCF")
#        if ft:
#            plt.legend()
#            ft=False
#        plt.pause(0.1)
#        l1.remove()
#        l2.remove()

    #This code plots the average over time
    #plt.plot(np.mean(DAV_profile_mean_total[:, :],0),'r-', label="DAV")
    #plt.plot(np.mean(TTCF_profile_mean_total[:, :],0),'b-', label="TTCF")
    #plt.legend()
    #plt.show()

    #Save variables at end of each batch in case of crash
    #np.savetxt('response_DAV.txt', DAV_response_mean_total)
    #np.savetxt('response_TTCF.txt', TTCF_response_mean_total)
    
    np.savetxt('profile_DAV.txt', DAV_profile_mean_total)
    np.savetxt('profile_TTCF.txt', TTCF_profile_mean_total)
    
    np.savetxt('profile_DAV_SE.txt', DAV_profile_var_total)
    np.savetxt('profile_TTCF_SE.txt', TTCF_profile_var_total)
    
    
    np.savetxt('response_DAV.txt', DAV_response_mean_total)
    np.savetxt('response_TTCF.txt', TTCF_response_mean_total)
    
    np.savetxt('response_DAV_SE.txt', DAV_response_var_total)
    np.savetxt('response_TTCF_SE.txt', TTCF_response_var_total)
    

MPI.Finalize()

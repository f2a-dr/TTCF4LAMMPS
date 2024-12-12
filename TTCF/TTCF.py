import numpy as np

from .utils import *

class TTCF():

    def __init__(self, global_variables, profile_variables, 
                 Nsteps, Nbins, Nmappings):

        """
            Create a TTCF object to collect data

            inputs include
            global_variables - list - variables collected by ave/time
            profile_variables - list - variables collected by ave/chunk
            Nsteps - int - Number of steps daughter us run for when TTCF is collected
            Nbins - int - Number of bins in profiles
            Nmappings - int - Number of mappings used by duaghters
        """

        self.global_variables = global_variables
        self.profile_variables = profile_variables
        self.Nsteps = Nsteps
        self.Nbins = Nbins
        self.Nmappings = Nmappings
        self.Count = 0

        avetime_ncol = len(global_variables)
        avechunk_ncol = len(profile_variables) + 2

        #Allocate arrays to store data
        self.DAV_global_mean  = np.zeros([Nsteps, avetime_ncol])
        self.DAV_profile_mean   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.DAV_global_var  = np.zeros([Nsteps, avetime_ncol])
        self.DAV_profile_var   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.DAV_profile_partial= np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.DAV_global_partial = np.zeros([Nsteps, avetime_ncol])

        # self.TTCF_global_var  = np.zeros([Nsteps, avetime_ncol])
        # self.TTCF_profile_var   = np.zeros([Nsteps, Nbins, avechunk_ncol])

        # self.TTCF_global_partial = np.zeros([Nsteps, avetime_ncol])
        # self.TTCF_profile_partial= np.zeros([Nsteps, Nbins, avechunk_ncol])

        # self.integrand_global_partial  = np.zeros([Nsteps, avetime_ncol])
        # self.integrand_profile_partial = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.global_partial_B_zero = np.zeros([Nsteps, avetime_ncol])
        self.global_partial_O_zero = np.zeros([Nsteps, avetime_ncol])
        self.global_partial_B = np.zeros([Nsteps, avetime_ncol])
        self.global_partial_OB = np.zeros([Nsteps, avetime_ncol])
        self.global_integrand_B = np.zeros([Nsteps,avetime_ncol])
        self.global_integrand_OB = np.zeros([Nsteps, avetime_ncol])
        self.global_B_zero_mean = np.zeros([Nsteps, avetime_ncol])
        self.global_O_zero_mean = np.zeros([Nsteps, avetime_ncol])
        self.global_B_mean = np.zeros([Nsteps, avetime_ncol])
        self.global_OB_mean = np.zeros([Nsteps, avetime_ncol])

        self.profile_partial_B_zero = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_partial_O_zero = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_partial_B = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_partial_OB = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_integrand_B = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_integrand_OB = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_B_zero_mean = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_O_zero_mean = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_B_mean = np.zeros([Nsteps, Nbins, avechunk_ncol])
        self.profile_OB_mean = np.zeros([Nsteps, Nbins, avechunk_ncol])

        self.TTCF_global_mean  = np.zeros([Nsteps, avetime_ncol])
        self.TTCF_profile_mean   = np.zeros([Nsteps, Nbins, avechunk_ncol])

    def add_mappings(self, data_profile, data_global, omega):

        #Sum the mappings together
        self.DAV_profile_partial  += data_profile[:,:,:]
        self.DAV_global_partial   += data_global[:,:]

        self.global_integrand_B += data_global[:,:]
        self.profile_integrand_B += data_profile[:,:,:]
        self.global_integrand_OB += data_global[:,:]*data_global[0,-1]
        self.profile_integrand_OB += data_profile[:,:,:]*data_global[0,-1]

        # Set initial values for Omega(t+=0) and B(t=0)
        # self.global_partial_B_zero[:,:] += data_global[0,:]
        self.profile_partial_B_zero[:,:,:] += data_profile[0,:,:]
        self.profile_partial_B_zero[:,:,3] = 0
        self.global_partial_O_zero[:,:] += data_global[0,-1]
        self.profile_partial_O_zero[:,:,:] += data_global[0,-1]


    def integration_setup(self, data_profile, data_global, omega):
        
        self.DAV_global_partial = data_global[:,:]
        self.DAV_profile_partial = data_profile[:,:,:]

        self.global_integrand_B = data_global[:,:]
        self.profile_integrand_B = data_profile[:,:,:]
        # self.global_integrand_OB = data_global[:,:]*omega
        # self.profile_integrand_OB = data_profile[:,:,:]*omega
        self.global_integrand_OB = data_global[:,:]*data_global[0,-1]
        self.profile_integrand_OB = data_profile[:,:,:]*data_global[0,-1]

        # Set initial values for Omega(t=0) and B(t=0)
        self.global_partial_B_zero[:,:] = data_global[0,:]
        self.profile_partial_B_zero[:,:,:] = data_profile[0,:,:]
        self.global_partial_O_zero[:,:] = data_global[0,-1]
        self.profile_partial_O_zero[:,:,:] = data_global[0,-1]

    def integrate(self, step, irank):

        # Perform the integration
        # self.TTCF_profile_partial = TTCF_integration(self.integrand_profile_partial, step)
        # self.TTCF_global_partial  = TTCF_integration(self.integrand_global_partial, step)

        # Integrate Omega(0)B(s)
        self.global_partial_OB = TTCF_integration(self.global_integrand_OB, step)
        self.profile_partial_OB = TTCF_integration(self.profile_integrand_OB, step)

        # Integrate B(s) for the correction term
        self.global_partial_B = TTCF_integration(self.global_integrand_B, step)
        self.profile_partial_B = TTCF_integration(self.profile_integrand_B, step)

        # Add the initial value (t=0) 
        # self.TTCF_profile_partial += self.DAV_profile_partial[0,:,:]
        # self.TTCF_global_partial  += self.DAV_global_partial[0,:]

        # Average over the mappings and update the Count (# of children trajectories generated excluding the mappings)
        self.DAV_profile_partial  /= self.Nmappings   
        self.DAV_global_partial   /= self.Nmappings 
        # self.TTCF_profile_partial /= self.Nmappings   
        # self.TTCF_global_partial  /= self.Nmappings 
        self.global_partial_B_zero /= self.Nmappings
        self.profile_partial_B_zero /= self.Nmappings
        self.global_partial_O_zero /= self.Nmappings
        self.profile_partial_O_zero /= self.Nmappings
        self.global_partial_OB /= self.Nmappings
        self.profile_partial_OB /= self.Nmappings
        self.global_partial_B /= self.Nmappings
        self.profile_partial_B /= self.Nmappings

        with open('PxyTime_' + str(irank+1) + '.dat', 'a') as f:
                # toWrite = str(self.global_partial_B_zero[:, 0] + self.global_partial_OB[:, 0] - self.global_partial_O_zero[:, 0]*self.global_partial_B[:, 0])
                toWrite = str(self.global_partial_OB[:, 0])
                f.write(toWrite)

        self.Count += 1

        # Update all means and variances
        if self.Count >1:

            # self.TTCF_profile_var= update_var(self.TTCF_profile_partial, self.TTCF_profile_mean, self.TTCF_profile_var, self.Count)      
            self.DAV_profile_var= update_var(self.DAV_profile_partial, self.DAV_profile_mean, self.DAV_profile_var, self.Count)
            # self.TTCF_global_var= update_var(self.TTCF_global_partial, self.TTCF_global_mean, self.TTCF_global_var, self.Count)   
            self.DAV_global_var= update_var(self.DAV_global_partial, self.DAV_global_mean, self.DAV_global_var, self.Count)
          
        # self.TTCF_profile_mean= update_mean(self.TTCF_profile_partial, self.TTCF_profile_mean, self.Count)     
        self.DAV_profile_mean= update_mean(self.DAV_profile_partial, self.DAV_profile_mean, self.Count)
        # self.TTCF_global_mean= update_mean(self.TTCF_global_partial, self.TTCF_global_mean, self.Count)
        self.DAV_global_mean= update_mean(self.DAV_global_partial, self.DAV_global_mean, self.Count)

        self.global_B_zero_mean = update_mean(self.global_partial_B_zero, self.global_B_zero_mean, self.Count)
        self.global_O_zero_mean = update_mean(self.global_partial_O_zero, self.global_O_zero_mean, self.Count)
        self.global_B_mean = update_mean(self.global_partial_B, self.global_B_mean, self.Count)
        self.global_OB_mean = update_mean(self.global_partial_OB, self.global_OB_mean, self.Count)
        
        self.profile_B_zero_mean = update_mean(self.profile_partial_B_zero, self.profile_B_zero_mean, self.Count)
        self.profile_O_zero_mean = update_mean(self.profile_partial_O_zero, self.profile_O_zero_mean, self.Count)
        self.profile_B_mean = update_mean(self.profile_partial_B, self.profile_B_mean, self.Count)
        self.profile_OB_mean = update_mean(self.profile_partial_OB, self.profile_OB_mean, self.Count)
        
        # np.savetxt('profile_B_zero_mean' + '_vx_' + str(self.Count) + '.txt', self.profile_B_zero_mean[:,:,2])
        # np.savetxt('profile_O_zero_mean' + '_vx_' + str(self.Count) + '.txt', self.profile_O_zero_mean[:,:,2])

        self.DAV_profile_partial[:,:,:] = 0
        self.DAV_global_partial[:,:]    = 0

        self.global_integrand_OB[:,:] = 0
        self.global_integrand_B[:,:] = 0
        self.profile_integrand_OB[:,:,:] = 0
        self.profile_integrand_B[:,:,:] = 0
            
        self.global_partial_B_zero[:,:] = 0
        self.profile_partial_B_zero[:,:,:] = 0
        self.global_partial_O_zero[:,:] = 0
        self.profile_partial_O_zero[:,:,:] = 0

    def finalise_output(self, irank, comm, root=0):

        self.irank = irank
        self.comm = comm
        self.root = root
        self.nprocs = comm.Get_size()
        self.output_finalised = True

        #Get FINAL COLUMN BECAUSE BY DEFAULT LAMMPS GIVE YOU ALSO THE USELESS INFO ABOUT THE BINS.
        # For  N QUANTITIES, could TAKE THE LAST N ELEMENTS
        self.profile_B_zero_mean = self.profile_B_zero_mean[:,:,2:]
        self.profile_O_zero_mean = self.profile_O_zero_mean[:,:,2:]
        self.profile_B_mean = self.profile_B_mean[:,:,2:]
        self.profile_OB_mean = self.profile_OB_mean[:,:,2:]

        # self.TTCF_profile_mean = self.TTCF_profile_mean[:,:,2:]
        self.DAV_profile_mean  = self.DAV_profile_mean[:,:,2:]

        # self.TTCF_profile_var = self.TTCF_profile_var[:,:,2:]
        self.DAV_profile_var  = self.DAV_profile_var[:,:,2:]

        # self.TTCF_global_var/= float(self.Count)
        self.DAV_global_var /= float(self.Count)
        # self.TTCF_profile_var /= float(self.Count)
        self.DAV_profile_var  /= float(self.Count)

        #Compute MEAN AND VARIANCE OF BOTH DAV AND TTCF
        # self.TTCF_profile_mean_total = sum_over_MPI(self.TTCF_profile_mean, irank, comm)
        self.DAV_profile_mean_total = sum_over_MPI(self.DAV_profile_mean, irank, comm)
        # self.TTCF_profile_var_total = sum_over_MPI(self.TTCF_profile_var, irank, comm)
        self.DAV_profile_var_total = sum_over_MPI(self.DAV_profile_var, irank, comm)

        # self.TTCF_global_mean_total = sum_over_MPI(self.TTCF_global_mean, irank, comm)
        self.DAV_global_mean_total = sum_over_MPI(self.DAV_global_mean, irank, comm)
        # self.TTCF_global_var_total = sum_over_MPI(self.TTCF_global_var, irank, comm)
        self.DAV_global_var_total = sum_over_MPI(self.DAV_global_var, irank, comm)

        self.global_B_zero_mean_total = sum_over_MPI(self.global_B_zero_mean, irank, comm)
        self.global_O_zero_mean_total = sum_over_MPI(self.global_O_zero_mean, irank, comm)
        self.global_B_mean_total = sum_over_MPI(self.global_B_mean, irank, comm)
        self.global_OB_mean_total = sum_over_MPI(self.global_OB_mean, irank, comm)

        self.profile_B_zero_mean_total = sum_over_MPI(self.profile_B_zero_mean, irank, comm)
        self.profile_O_zero_mean_total = sum_over_MPI(self.profile_O_zero_mean, irank, comm)
        self.profile_B_mean_total = sum_over_MPI(self.profile_B_mean, irank, comm)
        self.profile_OB_mean_total = sum_over_MPI(self.profile_OB_mean, irank, comm)

        #Total is None on everything but the root processor
        if irank == self.root:
            # self.TTCF_profile_mean_total = self.TTCF_profile_mean_total/float(self.nprocs)
            self.DAV_profile_mean_total  = self.DAV_profile_mean_total/float(self.nprocs)
            # self.TTCF_profile_var_total  = self.TTCF_profile_var_total/float(self.nprocs)
            self.DAV_profile_var_total   = self.DAV_profile_var_total/float(self.nprocs)
            
            # self.TTCF_global_mean_total = self.TTCF_global_mean_total/float(self.nprocs)
            self.DAV_global_mean_total  = self.DAV_global_mean_total/float(self.nprocs)
            # self.TTCF_global_var_total  = self.TTCF_global_var_total/float(self.nprocs)
            self.DAV_global_var_total   = self.DAV_global_var_total/float(self.nprocs)
            
            # self.TTCF_profile_SE_total  = np.sqrt(self.TTCF_profile_var_total)
            self.DAV_profile_SE_total   = np.sqrt(self.DAV_profile_var_total)
            # self.TTCF_global_SE_total   = np.sqrt(self.TTCF_global_var_total)
            self.DAV_global_SE_total    = np.sqrt(self.DAV_global_var_total)

            self.global_B_zero_mean_total = self.global_B_zero_mean_total/float(self.nprocs)
            self.global_O_zero_mean_total = self.global_O_zero_mean_total/float(self.nprocs)
            self.global_B_mean_total = self.global_B_mean_total/float(self.nprocs)
            self.global_OB_mean_total = self.global_OB_mean_total/float(self.nprocs)

            self.profile_B_zero_mean_total = self.profile_B_zero_mean_total/float(self.nprocs)
            self.profile_O_zero_mean_total = self.profile_O_zero_mean_total/float(self.nprocs)
            self.profile_B_mean_total = self.profile_B_mean_total/float(self.nprocs)
            self.profile_OB_mean_total = self.profile_OB_mean_total/float(self.nprocs)

            self.TTCF_global_mean_total = self.global_B_zero_mean_total + self.global_OB_mean_total - self.global_O_zero_mean_total*self.global_B_mean_total
            self.TTCF_profile_mean_total = self.profile_B_zero_mean_total + self.profile_OB_mean_total - self.profile_O_zero_mean_total*self.profile_B_mean_total

    def plot_data(self, animated=False):

        if self.output_finalised and self.irank == self.root:

            import matplotlib.pyplot as plt

            if animated:
                # This code animates the time history
                plt.ion()
                fig, ax = plt.subplots(1,1)
                plt.show()
                ft = True
                for t in range(self.TTCF_profile_mean_total.shape[0]):
                    print(t)
                    l1, = ax.plot(self.DAV_profile_mean_total[t, :],'r-', label="DAV")
                    l2, = ax.plot(self.TTCF_profile_mean_total[t, :],'b-', label="TTCF")
                    if ft:
                        plt.legend()
                        ft=False
                    plt.pause(0.1)
                    l1.remove()
                    l2.remove()
            else:
                #This code plots the average over time
                plt.plot(np.mean(self.DAV_profile_mean_total[:, :],0),'r-', label="DAV")
                plt.plot(np.mean(self.TTCF_profile_mean_total[:, :],0),'b-', label="TTCF")
                plt.legend()
                plt.show()

    def save_data(self):

        if self.output_finalised and self.irank == self.root:
            #print a separate file for each profile variable
            for i in range(len(self.profile_variables)):
            
                var_name=self.profile_variables[i]
                #replace "/" character in the name to avoid crashes (it is read as a folder)
                var_name=var_name.replace('/', '_')
             
                print(var_name)
                
                np.savetxt('profile_DAV_' + var_name + '.txt', self.DAV_profile_mean_total[:,:,i])
                np.savetxt('profile_TTCF_' + var_name + '.txt', self.TTCF_profile_mean_total[:,:,i])
                np.savetxt('profile_DAV_SE_' + var_name + '.txt', self.DAV_profile_SE_total[:,:,i])
                # np.savetxt('profile_B_zero_mean_total' + var_name + '.txt', self.profile_B_zero_mean_total[:,:,i])
                # np.savetxt('profile_O_zero_mean_total' + var_name + '.txt', self.profile_O_zero_mean_total[:,:,i])
                # np.savetxt('profile_B_mean_total' + var_name + '.txt', self.profile_B_mean_total[:,:,i])
                # np.savetxt('profile_OB_mean_total' + var_name + '.txt', self.profile_OB_mean_total[:,:,i])
                # np.savetxt('profile_TTCF_SE_' + var_name + '.txt',self.TTCF_profile_SE_total[:,:,i])
            
            
            #np.savetxt('profile_DAV.txt', self.DAV_profile_mean_total)
            #np.savetxt('profile_TTCF.txt', self.TTCF_profile_mean_total)
            
            #np.savetxt('profile_DAV_SE.txt', self.DAV_profile_SE_total)
            #np.savetxt('profile_TTCF_SE.txt', self.TTCF_profile_SE_total)
            
            # np.savetxt('global_B_zero_mean_total.txt', self.global_B_zero_mean_total)
            # np.savetxt('global_O_zero_mean_total.txt', self.global_O_zero_mean_total)
            # np.savetxt('global_B_mean_total.txt', self.global_B_mean_total)
            # np.savetxt('global_OB_mean_total.txt', self.global_OB_mean_total)
            np.savetxt('global_DAV.txt', self.DAV_global_mean_total)
            np.savetxt('global_TTCF.txt', self.TTCF_global_mean_total)
            
            np.savetxt('global_DAV_SE.txt', self.DAV_global_SE_total)
            # np.savetxt('global_TTCF_SE.txt', self.TTCF_global_SE_total)
                 


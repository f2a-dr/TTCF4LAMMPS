#! /usr/bin/env python2.7
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/home/es205/codes/python/SimWrapPy/')
import simwraplib as swl

def ms(s):
    return str(-float(s))

def phase_space_map(restartfile, maptype="flipymompos"):
    nl = 0; nlv = 0
    with open(restartfile, "r") as f:
        with open("mirror_"+restartfile, "w+") as g:
            for l in f:
                #Get number of atoms
                if "atoms" in l:
                    N = int(l.replace("atoms",""))
                #For Atoms, rewrite the next N records 
                if "Atoms" in l and "pos" in maptype:
                   nl = N
                #For velocities, rewrite the next N records 
                if "Velocities" in l:
                    nlv = N
                #Flip domain extents as well
                if "flipymompos" in maptype and "ylo" in l:
                    a = l.split()
                    g.write(   ms(a[1]) + " " + ms(a[0]) + " " 
                            + a[2] + " " + a[3] + "\n" )
                #If next line (nlv) is not zero, adapt/write this line
                elif nl != 0:
                    #Error handling here to skip any non-velocity records
                    try:
                        a = l.split()
                        if "flipymompos" in maptype:
                            g.write(  a[0] + " " + a[1]+ " " 
                                    + a[2] + " " + a[3]+ " " 
                                    + a[4]+ " " + ms(a[5]) + " " 
                                    + a[6] + " " + a[7] + " " 
                                    + a[8] + " " + a[9] + "\n")
                        nl -= 1
                    except IndexError:
                        g.write(l)

                #If next line (nlv) is not zero, adapt/write this line
                elif nlv != 0:
                    #Error handling here to skip any non-velocity records
                    try:
                        a = l.split()
                        if "reflectmom" in maptype:
                            g.write(a[0] + " " + ms(a[1]) 
                                         + " " + ms(a[2])
                                         + " " + ms(a[3]) + "\n")
                        elif "flipymompos" in maptype:
                            g.write(a[0] + " " + a[1]
                                         + " " + ms(a[3])
                                         + " " + a[2] + "\n")
                        nlv -= 1
                    except IndexError:
                        g.write(l)
                else:
                    g.write(l)


#Directory which executable and input files are found
fdir = "/home/es205/scratch/TTCF/"
basedir = fdir + "/couette/"

#Number of processors to use
ncpus = 6

#Directory used to compile executale (also used to archive details of run for future generations)
srcdir =  "/home/es205/codes/cpl_granlammmps/OpenFOAM-3.0.1_LAMMPS-dev/LAMMPS-dev_coupled/"

#Setup changes to make to input file
restartfile = "output{:d}.dat"
children = []
#for i in range(100300, 303300, 300):
for i in range(100300, 100900, 300):
    children.append(restartfile.format(i))
    children.append("mirror_" + restartfile.format(i))

changeDict = swl.InputDict({'read_data': children})
changes = changeDict.expand()
filenames = changes.filenames()

#Loop over all changes and assign each to a thread
threadlist =[]
for thread in range(0,len(changes)):

    rundir = basedir + "ttcf" + filenames[thread].replace("readdata","").replace("output","").replace("pdat","")

    rfile = changes[thread]["read_data"]
    if (os.path.isfile(rfile)):
        #Create corresponding mirror file
        if not "mirror" in rfile:
            phase_space_map(rfile)

        run = swl.LammpsRun(srcdir,
                            basedir,
                            rundir,
                            executable='./lmp_cpl', #Relative in basedir
                            inputfile='walls_imaginary_sliding_read_bulk.in', #Relative to basedir
                            outputfile='lammps.out', #Relative to basedir
                            restartfile=rfile,
                            deleteoutput=True)
                            #inputchanges=changes[thread])

        #One run for this thread (i.e. no setup run before main run)
        #run.finish = finish

        runlist = [run]
        threadlist.append(runlist)
        print('Run in directory '  + rundir + 
               ' and dryrun is '  + str(run.dryrun))
    else:
        print("Restart file ", changes[thread]["read_data"], " not found")

#Run the study which contains all threads
study = swl.Study(threadlist, ncpus, studyfolder="study")

#Run mother trajectory




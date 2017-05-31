Minimal version of Forecasted dynamic Theil's entropy software:
To run the program you need Python 2.7.x, PyQt4, matplotlib, 
numpy and scipy. 

The code has been tested and developed under Linucx, but once installed the needed 
packages should work also under Mac OS X and Windows OS.

To run the GUI:

python markovc_qt.py 

to run the CLI:

python markovc.py

usage: markovc.py [-h] -m RMATFILENAME -b IMATFILENAME [-s STEP] [-t TPREV] -n
                  MAXRUN [-M NAMEOFMATRIX] [-B NAMEOFBPMATRIX] [-v] [-i] [-S]

optional arguments:
  -h, --help            show this help message and exit
  -m RMATFILENAME, --rmat-filename RMATFILENAME
                        Transition probability matrix filename
  -b IMATFILENAME, --imat-filename IMATFILENAME
                        Rewards matrix filename
  -s STEP, --step STEP  Bin width
  -t TPREV, --time-prev TPREV
                        Forecasted period
  -n MAXRUN, --max-run MAXRUN
                        Monte carlo iterations
  -M NAMEOFMATRIX, --name-of-matrix NAMEOFMATRIX
                        Name of the probability matrix
  -B NAMEOFBPMATRIX, --name-of-bpmatrix NAMEOFBPMATRIX
                        Name of the rewards matrix
  -v, --verbose         increase output verbosity
  -i, --time-inf        Simulation using stationary distribution
  -S, --seed            Using a seed for the random generator



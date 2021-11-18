Minimal version of Forecasted dynamic Theil's entropy software:
To  run the program you need Python 3, PyQt5, matplotlib, numpy 
and scipy. 

The code has been  tested and developed under Linux,  but  once 
installed  the needed  packages should work also under Mac OS X 
and Windows OS.

To run the GUI:

python randentropy_qt.py

to run the GUI you may need to export QT_X11_NO_MITSHM=1

to run the CLI:

python randentropy.py

Finally you'll find also a changepoint.py CLI

Quick  start:  to  perform some test using the changepoint CLI 
you can use the input files in ./files specifically:

$ python3 changepoint.py -m ./files/sep_monthly.mat -c 1

you should  get  Change Point:  158  ( -320.0611462871186 ) as
a result. Similarly you can run the same using the GUI.

To use two change-points:

$ python3 changepoint.py -m ./files/sep_monthly.mat -c 2 \
	--cp2-start 50 --cp2-stop 157

you should get the following results:

Change Point:  73  ,  119  ( -311.1237648393643 )

You can run also the Lambda test:

$ python3 changepoint.py -m ./files/sep_monthly.mat -c 2 \
	--perform-test "73;119;100"

To  test the  randentropy CLI,  you  can  use the same file as 
before:

$ python3 randentropy.py -m ./files/sep_monthly.mat \
  -b ./files/sep_monthly.mat -s 0.25 -t 36 -n 1000 -v

the sep_monthly.mat contains both  the  community as  well  as
the     attributes      matrices.    The     same      results 
(i.e. entropy_1000.txt in this case) can be obtained using the 
GUI (i.e. randentropy_qt.py).

Cite this as:

D’Amico G, Scocchera S, Storchi L (2018b). “Financial risk distribution in European Union.”
Physica A: Statistical Mechanics and its Applications, 505, 252–267.

D’Amico G, Petroni F, Regnault P, Scocchera S, Storchi L (2019). “A Copula-based Markov
Reward Approach to the Credit Spread in the European Union.” Applied Mathematical
Finance, 26(4), 359–386.

D’Amico G, Regnault P, Scocchera S, Storchi L (2018a).  
“A Continuous-Time InequalityMeasure Applied to Financial Risk:  The Case of the European Union.”
International Journal of Financial Studies, 6(3), 62.

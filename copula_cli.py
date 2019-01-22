import scipy.stats
import matplotlib.pyplot as plt
import argparse
import scipy.io
import numpy
import math
import sys
import os

import os.path

sys.path.append("./module")
import basicutils

import matplotlib.pyplot as plt

#######################################################################
# adapted from
# https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one

def ecdf(raw_data):
    
    data = numpy.asarray(raw_data)
    data = numpy.atleast_1d(data)
    quantiles, counts = numpy.unique(data, return_counts=True)
    cumprob = numpy.cumsum(counts).astype(numpy.double) / data.size

    return quantiles, cumprob

#######################################################################


parser = argparse.ArgumentParser()

parser.add_argument("-m","--mat-filename", help="Transition probability matrix filename", \
        type=str, required=True, dest="matfilename")
parser.add_argument("-R", "--name-of-rating-matrix", help="Name of the rating matrix ", \
        type=str, required=False, default="ratings", dest="nameofratingmatrix")
parser.add_argument("-S", "--name-of-spread-matrix", help="Name of the spread matrix ", \
        type=str, required=False, default="spread", dest="nameofspreadmatrix")
parser.add_argument("-P", "--name-of-prating-matrix", help="Name of the rating matrix ", \
        type=str, required=False, default="p_rating", dest="nameofpratingmatrix")
parser.add_argument("-v", "--verbose", help="increase output verbosity", \
        default=False, action="store_true")

if len(sys.argv) == 1:
    parser.print_help()
    exit(1)

args = parser.parse_args()

name_rtmt = args.nameofratingmatrix
name_spmt = args.nameofspreadmatrix
name_prmt = args.nameofpratingmatrix
filename = args.matfilename
verbose = args.verbose

errmsg = []

if not (os.path.isfile(filename)):
    print ("File " + filename + " does not exist ")
    exit(1)

matf = scipy.io.loadmat(filename)

if not(name_rtmt in matf.keys()):
    print "Cannot find " + name_rtmt + " in " + filename
    print matf.keys()
    exit(1)

if not(name_spmt in matf.keys()):
    print "Cannot find " + name_spmt + " in " + filename
    print matf.keys()
    exit(1)

ratings = matf[name_rtmt]
spread = matf[name_spmt]
p_rating = matf[name_prmt]
#print ratings.shape
#print spmt.shape
#for i in range(ratings.shape[0]):
#    for j in range(ratings.shape[1]):
#        print "%3d"%ratings[i, j]


if ratings.shape != spread.shape:
    print "Error  in matrix dimension"
    exit(1)

N = numpy.max(ratings)
Nnaz = spread.shape[0]
Dst = max(spread.shape)

#print "N: ", N, "Nnaz: " , Nnaz, "Dst: ", Dst

inc_spread = numpy.zeros((Nnaz,Dst-1))

end = spread.shape[1]
for i in range(Nnaz):
    a = spread[i,1:end] - spread[i,0:end-1]
    b = spread[i,0:end-1]
    inc_spread[i,:] = numpy.divide(a, b, out=numpy.full_like(a, float("Inf")), where=b!=0)

rttmp = ratings[:,1:end]
totdim = rttmp.shape[0]*rttmp.shape[1]

rttmp = rttmp.reshape(totdim, order='F')
f_inc_spread = inc_spread.reshape(totdim, order='F')

#for i in f_inc_spread:
#    print "%10.5f"%(i)

start_spread_for_rating = []
for ratclass in range(1,8+1):
    start_spread_for_rating.append([])
    for i in range(Nnaz):
#        if (i != 3):
          for j in range(ratings.shape[1]):
              if ratings[i, j] == ratclass:
                  start_spread_for_rating[ratclass-1].append(spread[i, j])

    s = numpy.std(start_spread_for_rating[ratclass-1])
    m = numpy.mean(start_spread_for_rating[ratclass-1])
                
    print len(start_spread_for_rating[ratclass-1]), m, s

spread_for_rating = []
for ratclass in range(1,8+1):
    s = numpy.std(start_spread_for_rating[ratclass-1])
    m = numpy.mean(start_spread_for_rating[ratclass-1])

    spread_for_rating.append([])

    #spread_for_rating[ratclass-1] = \
    #        [v for v in start_spread_for_rating[ratclass-1]]

    #spread_for_rating[ratclass-1] = \
    #        [v for v in start_spread_for_rating[ratclass-1] if \
    #        (math.fabs(v-m) <= 3.0*s and v != 0.0) ]

    spread_for_rating[ratclass-1] = \
        [v for v in start_spread_for_rating[ratclass-1] if \
        (v > 0.0) ]



    s = numpy.std(spread_for_rating[ratclass-1])
    m = numpy.mean(spread_for_rating[ratclass-1])

    #print len(spread_for_rating[ratclass-1]), m, s


for ratclass in range(1,8+1):
    #m = numpy.mean(spread_for_rating[ratclass-1])
    #for i in range(len(spread_for_rating[ratclass-1])):
    #    spread_for_rating[ratclass-1][i] -= m
    s = numpy.std(spread_for_rating[ratclass-1])
    m = numpy.mean(spread_for_rating[ratclass-1])

    fp1 = open("spread"+str(ratclass)+".txt", "w")
    fp2 = open("lognorm"+str(ratclass)+".txt", "w")
    fp3 = open("spread_cdf"+str(ratclass)+".txt", "w")
    fp4 = open("lognorm_cdf"+str(ratclass)+".txt", "w")

    #y, x = numpy.histogram(spread_for_rating[ratclass-1], 100, normed=True)
    params = scipy.stats.lognorm.fit(spread_for_rating[ratclass-1])
    params = scipy.stats.lognorm.fit(spread_for_rating[ratclass-1], params[0], loc=params[1], scale=params[2])
    params = scipy.stats.lognorm.fit(spread_for_rating[ratclass-1], params[0], loc=params[1], scale=params[2])
 
    print scipy.stats.kstest(spread_for_rating[ratclass-1], "lognorm", params, 
            alternative="two-sided", mode="asymp")

    data_sorted = numpy.sort(spread_for_rating[ratclass-1])
    p = 1. * numpy.arange(len(spread_for_rating[ratclass-1])) / \
            (len(spread_for_rating[ratclass-1]) - 1)

    for i in range(len(data_sorted)):
        fp3.write(str(data_sorted[i]) + " " + str(p[i]) + "\n")


    """
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(p, data_sorted)
    ax1.set_xlabel('$p$')
    ax1.set_ylabel('$x$')

    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')

    plt.show()
    """

    count, bins, ignored = plt.hist(spread_for_rating[ratclass-1], 100, \
            normed=True, align="mid")
    for i in range(len(count)):
        fp1.write(str((bins[i]+bins[i+1])/2.0)  + " " + str(count[i]) + "\n")

    x=numpy.linspace(min(bins),max(bins),1000)
    pdf = scipy.stats.lognorm.pdf(x, params[0], loc=params[1], scale=params[2])
    for i in range(len(x)):
        fp2.write(str( x[i]) + " " + str(pdf[i]) + "\n")

    cdf = scipy.stats.lognorm.cdf(data_sorted, params[0], loc=params[1], scale=params[2])
    for i in range(len(data_sorted)):
        fp4.write(str( data_sorted[i]) + " " + str(cdf[i]) + "\n")
        
    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()

for ratclass in range(1,8+1):

    log_spread_for_rating = numpy.log(spread_for_rating[ratclass-1])

    s = numpy.std(log_spread_for_rating)
    m = numpy.mean(log_spread_for_rating)

    value = [((v-m)/s) for v in log_spread_for_rating]

    print scipy.stats.jarque_bera (value)

 

exit(1)


X = []
G = []

for i in range(N):
    tmp = numpy.where(rttmp == i+1)[0]
    dist_sp = [f_inc_spread[j] for j in tmp]
    dist_sp = filter(lambda a: a != float("Inf"), dist_sp)
    mind = scipy.stats.mstats.mquantiles(dist_sp, 0.05)
    maxd = scipy.stats.mstats.mquantiles(dist_sp, 0.95)

    dist_sp = filter(lambda a: a >= mind and a <= maxd, dist_sp)

    x, y = ecdf(dist_sp)
    X.append(x)
    G.append(y)
    #basicutils.vct_to_stdout(numpy.asarray(G[i]))
    #basicutils.vct_to_stdout(numpy.asarray(dist_sp))
    #basicutils.vct_to_stdout(numpy.asarray(X[i]))
    #for j in range(len(G[i])):
    #    print "%10.5f %10.5f"%(X[i][j], G[i][j]) 

d = 365*3
Nsim = 200
rho = numpy.corrcoef(spread) # don't need to transpose 

#for i in range(rho.shape[0]):
#    for j in range(rho.shape[1]):
#        print "%10.5f "%(rho[i,j])

exit()

"""

%% montecarlo rating e spread
tau = copulastat('gaussian',rho);
R_in = rating(:,end);
spread_synth_tmp = zeros(1,Nnaz);
spread_synth = zeros(Nsim,d,Nnaz);

entropy_t = .4.*ones(Nsim,d);
for sim = 1:Nsim
    spread_synth(sim,1,:) = spread(:,end)'+eps;
    u = copularnd('gaussian',rho,d); 
    for j=2:d
        v = rand(1,Nnaz);
        pp = cumsum(P_rating(R_in,:),2);
        jj = zeros(1,Nnaz);
        for k=1:Nnaz
            jj(1,k)=find(pp(k,:)>=v(1,k),1,'first');
            spread_synth_tmp(1,k) = max(interp1(G{jj(1,k)}(2:end),X{jj(1,k)}(2:end),u(j,k),'linear','extrap'),-0.9); %quantile(G{jj(1,k)},u(j,k));
        end
        R_in = jj;
        spread_synth(sim,j,:) = squeeze(spread_synth(sim,j-1,:)).*(1+spread_synth_tmp(1,:)');
        %% entropia %%
        P_spread = spread_synth(sim,j,:)./sum(spread_synth(sim,j,:));%+0.00001;
        entropy_t(sim,j) =  sum(P_spread.*log(Nnaz.*P_spread));   
    end
end


%%
figure
plot(mean(entropy_t(:,2:end),1))
title('Entropy')
figure
plot(squeeze(spread_synth(3,:,:)))
save(['Spread_synth_',agenzia],'spread_synth');
%%
save(['entropy_',agenzia],'entropy_t')
"""

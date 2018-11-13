import scipy.io
import argparse
import numpy
import sys
import os

import os.path

sys.path.append("./module")
import basicutils


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
#print rtmt.shape
#print spmt.shape

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
    inc_spread[i,:] = numpy.divide(a, b, out=numpy.zeros_like(a), where=b!=0)


#for i in range(inc_spread.shape[0]):
#    for j in range(inc_spread.shape[1]):
#        print inc_spread[i, j]

for i in range(N):
    tmp = numpy.argwhere(ratings[:,1:] == i+1)
    for j in tmp:
        print j[0], j[1]
        #indexes.append(j[0])

    #v = numpy.argwhere(numpy.abs(inc_spread - 0.13831) < 0.00001)
    #for j in v:
    #    print j
    #    print inc_spread[tuple(j)]
    #print ""
  
    #for j in tmp:
        #print j
        #print ratings[tuple(j)]
    #    print "%10.5f"%(inc_spread[j[0]+1, j[1]])

    dist_sp = [inc_spread[tuple(j)] for j in tmp]
    #for va in dist_sp:
    #    print "%10.5f"%(va)
    exit()

"""

for i=1:N
    tmp = find(rating(:,2:end) ==i);
    dist_sp = inc_spread(tmp);
    dist_sp(dist_sp==inf)=[];
    dist_sp(dist_sp== -inf)=[];
    dist_sp(isnan(dist_sp)==1)=[];
    dist_sp(isnan(dist_sp)==-1)=[];
    a = quantile(dist_sp,.05);
    b = quantile(dist_sp,.95);
    dist_sp(dist_sp<a | dist_sp>b)=[];
    [G{i},X{i}] = ecdf(dist_sp);
end


%% montecarlo rating e spread
d = 365*3;
Nsim = 200;
rho = corrcoef(spread'); % covariance matrix
%rho = corrcoef(spread','Rows','complete');
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

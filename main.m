clear
tic
load ms.mat;
time=length(ms(1,:));
Nk=zeros(8,8,26);
Num=zeros(8,8);
Den=zeros(8,1);
Pr=zeros(8,8);
for k=1:26
    for time=1:226
        for i=1:8
            for j=1:8
                if ms(k,time)==i && ms(k,time+1)==j
                    Nk(i,j,k)=Nk(i,j,k)+1;
                end
                Num(i,j)=sum(Nk(i,j,:));
            end

            Den(i)=sum(Num(i,:));
        end
    end
    printf ("%d of 26\n", k )
    fflush(stdout);
end

for i=1:8
    for j=1:8
        if Den(i)~=0
            Pr(i,j)=Num(i,j)/Den(i);
        else Pr(i,j)=0;
        end
    end
end

mPr=mean(svd(Pr-eye(size(Pr))));

%------Spreads----------------
%per calcolare i bp abbiamo scaricato i tassi di interesse pagati sui
%titoli sovrani a 10 anni. i dati sono raccolto con frequenza mensile 
%così come i rating.
load bp.mat;
benchmark=min(i_r);
k=length(i_r(:,1));   %numero paesi 
Time=length(i_r(1,:));    % orizzonte temporale 
r=zeros(26,227);
for k=1:26
   for Time=1:227
       r(k,Time)=i_r(k,Time)-benchmark(Time);
    end
end

ist=zeros(8,227*26);% istogramma dello spread in funzione della classe di rating. ogni vettore rappresenta tutti i valori dello spread quando ho un determinato rating. 
Nn=zeros(8,1); %conta quante volte viene registrato un determinato rating. 
for i=1:8
    for k=1:26
        for Time=1:227
            if ms(k,Time)==i; %quando incontro un determinato rating i dati di input
                Nn(i) = Nn(i)+1; %il vettore aggiunge +1
                ist(i,Nn(i)) = r(k,Time);% mentre l'istogramma aggiunge al vettore del rating i il valore corrispondente dello spread
            end
        end
    end
end

Mean=zeros(1,7);
y = ist(:,1:2135);% mi toglie tutte le righe con gli 0
aaa = y(1,:);
aaa = aaa(isfinite(aaa));
aa = y(2,1:879);
aa = aa(isfinite(aa));
a = y(3,1:1248);
a = a(isfinite(a));
bbb = y(4,1:1033);
bbb = bbb(isfinite(bbb));
bb = y(5,1:410);
bb = bb(isfinite(bb));
b = y(6,1:131);
b=b(isfinite(b));
cc=y(7,1:66);
cc=cc(isfinite(cc));
d=y(8,1:1);

AAA = hist(aaa,0:0.25:2.82);
Paaa = AAA/sum(AAA);
AA = hist(aa,0:0.25:5.69);
Paa = AA/sum(AA);
A = hist(a,0:0.25:11.57);
Pa = A/sum(A);
BBB = hist(bbb,0.13:0.25:11.29);
Pbbb = BBB/sum(BBB);
BB = hist(bb,1.52:0.25:12.15);
Pbb = BB/sum(BB);
B = hist(b,9.23:0.25:12.93);
Pb = B/sum(B);
CC = hist(cc,4.67:0.25:27.40);
Pcc = CC/sum(CC);

Taaa=sum(Paaa.*(log(length(Paaa))*Paaa));
Taa=sum(Paa.*(log(length(Paa))*Paa));
Ta=sum(Pa.*(log(length(Pa))*Pa));
Tbbb=sum(Pbbb.*(log(length(Pbbb))*Pbbb));
Tbb=sum(Pbb.*(log(length(Pbb))*Pbb));
Tb=sum(Pb.*(log(length(Pb))*Pb));
Tcc=sum(Pcc.*(log(length(Pcc))*Pcc));

Mean=[mean(aaa), mean(aa),mean(a),mean(bbb), mean(bb),mean(b),mean(cc)];
Ti=[Taaa, Taa, Ta, Tbbb, Tbb, Tb, Tcc];
#MEANST=[mean(aaa), mean(aa),mean(a),mean(bbb), mean(bb),mean(b),mean(cc); std(aaa),std(aa), std(a), std(bbb), std(bb), std(b), std(cc)];
#minmax=[min(aaa) max(aaa); min(aa) max(aa); min(a) max(a); min(bbb) max(bbb); min(bb) max(bb); min(b) max(b); min(cc) max(cc); min(d) max(d)];

# here

s_t=zeros(26,227);
r(isnan(r))=0;
R_t=sum(r);
T_t=zeros(1,227);
countries=26;
 for t=1:227
     for k=1:26
        s_t(k,t)=r(k,t)/R_t(t);
        if s_t(k,t)~=0
        T_t(t)=T_t(t)+s_t(k,t)*log(countries*s_t(k,t));
        end
    end
 end

%----previsione per 36 mesi fino a 11/2019 con simulazione montecarlo e
%formula PDTE del paper D'Amico et al.

run=1000;
X=rand(26,37,run);% per simulare le traiettorie
x=zeros(26,37,run);%possibile assegnazione rating in base  Pr
cdf=zeros(8,8);%pdf 
bp=zeros(26,37,run);% basis point assegnati secondo il rating
xm=zeros(26,37);
r_prev=zeros(1,37,run);% tot punti base pagati da tutti i paesi
Var=zeros(1,37); % II momento dell'entropia
tot=zeros(7,37);  %basis point pagati dalla classe di rating i
cont=zeros(7,37,run); %numero di paesi 
ac=zeros(7,37,run);%quota di bp pagata dalla classe di rating i
AC=zeros(7,37);
t1=zeros(1,37,run);% I membro equazione PDTE
t2=zeros(1,37,run);% II membro equazione PDTE
term=zeros(1,37,run);%III membro equazione PDTE
entr=zeros(1,37,run);%PDTE
entropia=zeros(1,37);
R_prev=zeros(1,37);

save('XcdfPr.mat', 'X', 'cdf', 'Pr', '-mat7-binary')

for j=1:run
    x(:,1,j)=ms(:,227,:);

    for i=1:8
        cdf(i,1)=Pr(i,1);
    end
    for i=1:8
        for k=2:8
            cdf(i,k)=Pr(i,k)+cdf(i,k-1);
        end
    end

    for c=1:26
        if X(c,1,j)<=cdf(x(c,1,j),1)
            x(c,2,j)=1;
        end
        for k=2:8
            if cdf(x(c,1,j),k-1)<X(c,1,j) && X(c,1,j)<=cdf(x(c,1,j),k)
                x(c,2,j)=k;
            end
        end
        for t=3:37
            if X(c,t-1,j)<=cdf(x(c,t-1,j),1)
                x(c,t,j)=1;
            end
            for k=2:8
                if cdf(x(c,t-1,j),k-1)<X(c,t-1,j) && X(c,t-1,j)<=cdf(x(c,t-1,j),k)
                  x(c,t,j)=k;
                end
            end
        end
    end  

    for t=1:37
        for c=1:26
            for i=1:7
                if x(c,t,j)==i;
                    bp(c,t,j)=Mean(1,i);
                    cont(i,t,j)=cont(i,t,j)+1;
                    tot(i,t,j)=cont(i,t,j).*Mean(1,i);
                end
            end
            r_prev(1,t,j)=sum(bp(:,t,j));
        end
    end

    for t=1:37
        for i=1:7
             ac(i,t,j)=tot(i,t,j)/r_prev(1,t,j);
             if ac(i,t,j)~=0
             t1(1,t,j)=t1(1,t,j)+(ac(i,t,j).*Ti(i));    
             t2(1,t,j)=t2(1,t,j)+ac(i,t,j)*log(7*ac(i,t,j));
                if cont(i,t,j)~=0;
                  term(1,t,j)=term(1,t,j)+ac(i,t,j).*log(countries/(7*cont(i,t,j)));
                end
             end  
             AC(i,t)=mean(ac(i,t,:));
        end
        entr(1,t,j)=t1(1,t,j)+t2(1,t,j)+term(1,t,j);
        R_prev(1,t)=mean(r_prev(1,t,j));
    end

    printf ("%d of %d\n", j, run);
    fflush(stdout);
end  

for t=1:37
    entropia(1,t)=mean(entr(1,t,:));
    printf ("%d %f\n", t, entropia(1,t));
    Var(1,t)=std(entr(1,t,:));
    #disp(Var(1,t));
end


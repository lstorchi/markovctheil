from PyQt5 import QtGui, QtCore, QtWidgets

import plothist
import scipy.io
import options
import matwin
import numpy
import os

import os.path

import basicutils

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import mkvtmod

class main_window(QtWidgets.QMainWindow):

    def __init__(self):

        self.__inputfile__ = ""
        self.__fileio__ = False
        self.__entropiadone__ = False
        self.__plot_done__ = False

        self.__entropy__ = None
        self.__inter_entropy__ = None
        self.__intra_entropy__ = None
        self.__var__ = None
        self.__allratings__ = None
        self.__allratingsnins__ = None
        self.__meanval__ = None
        self.__stdeval__ = None
        self.__pr__ = None
       
        QtWidgets.QMainWindow.__init__(self) 
        self.resize(640, 480) 
        self.setWindowTitle('Dynamic Theil entropy')
        self.statusBar().showMessage('Markov started') 

        ofile = QtWidgets.QAction(QtGui.QIcon("icons/open.png"), "Open", self)
        ofile.setShortcut("Ctrl+O")
        ofile.setStatusTip("Open file")
        ofile.triggered.connect(self.openfile)

        self.__savefile__ = QtWidgets.QAction(QtGui.QIcon("icons/save.png"), "Save", self)
        self.__savefile__.setStatusTip("Save file")
        self.__savefile__.triggered.connect(self.savefile)
        self.__savefile__.setEnabled(False)

        sep = QtWidgets.QAction(self)
        sep.setSeparator(True)

        quit = QtWidgets.QAction(QtGui.QIcon("icons/cancel.png"), "Quit", self)
        quit.setShortcut("Ctrl+Q")
        quit.setStatusTip("Quit application")
        quit.triggered.connect(self.close)

        run = QtWidgets.QAction(QtGui.QIcon("icons/run.png"), "Run", self)
        run.setShortcut("Ctrl+R")
        run.setStatusTip("Run")
        run.triggered.connect(self.mainrun)

        runchangepoint = QtWidgets.QAction(QtGui.QIcon("icons/run.png"), "RunChangePoint", self)
        runchangepoint.setStatusTip("Run ChangePoint")
        runchangepoint.triggered.connect(self.runchangepoint)

        self.__plots__ = QtWidgets.QAction(QtGui.QIcon("icons/save.png"), "Plot Empirical distributions", self)
        self.__plots__.setShortcut("Ctrl+S")
        self.__plots__.setStatusTip("Credit spread distributions")
        self.__plots__.triggered.connect(self.plot_hist)
        self.__plots__.setEnabled(False)

        self.__mats__ = QtWidgets.QAction(QtGui.QIcon("icons/save.png"), "View transition matrix", self)
        self.__mats__.setShortcut("Ctrl+M")
        self.__mats__.setStatusTip("View transition matrices")
        self.__mats__.triggered.connect(self.view_mats)
        self.__mats__.setEnabled(False)

        self.statusBar().show()

        menubar = self.menuBar()
        
        file = menubar.addMenu('&File')
        file.addAction(ofile)
        file.addAction(self.__savefile__)
        file.addAction(sep)
        file.addAction(quit)

        edit = menubar.addMenu('&Edit')
        edit.addAction(run)
        edit.addAction(runchangepoint)
        edit.addAction(self.__plots__ )
        edit.addAction(self.__mats__ )

        help = menubar.addMenu('&Help')

        self.__figure__ = plt.figure()
        self.__canvas__ = FigureCanvas(self.__figure__)
        self.__toolbar__ = NavigationToolbar(self.__canvas__, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.__toolbar__)
        layout.addWidget(self.__canvas__)

        maindialog = QtWidgets.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)

        self.__options_dialog__ = options.optiondialog(self)

        self.__options_dialog_cp__ = options.optiondialog_cp(self)

        self.__options_name_dialog__ = options.optionnamedialog(self)

    def savefile(self):

        if self.__entropiadone__ :
           tosave = QtWidgets.QFileDialog.getSaveFileName(self) 
           if os.path.exists(str(tosave)):
               os.remove(str(tosave))
               
           outf = open(str(tosave), "w")
           
           outf.write("X ")
           for i in range(0, len(self.__meanval__)):
               outf.write("K=%d "%(i+1))
           outf.write("\n")

           outf.write("E(BP) ")
           for m in self.__meanval__:
               outf.write("%f "%(m))
           outf.write("\n")

           outf.write("STD(BP) ")
           for m in self.__stdeval__:
               outf.write("%f "%(m))
           outf.write("\n")

           outf.write("\n")

           if self.__options_dialog__.getinftime():
                ev = self.__entropy__[1:]
                m = numpy.mean(ev)
                s = numpy.std(ev)
                outf.write("Stationary value: " + \
                        str(m) + " stdev: " + str(s))
           else:
                outf.write("t E(DT) STD(DT)\n")
                for t in range(self.__options_dialog__.gettprev()):
                    outf.write("%d %f %f \n"%(t+1, self.__entropy__[t], \
                            self.__var__[t]))

           outf.close()

    def openfile(self):

        self.__inputfile__ = QtWidgets.QFileDialog.getOpenFileName(self, 
                "Open File", "./", "Mat or CSV (*.mat *.csv)")[0]

        self.__fileio__ = False
        
        self.__entropiadone__ = False
        self.__savefile__.setEnabled(False)
        
        self.__plots__.setEnabled(False)
        self.__mats__.setEnabled(False)

        if self.__plot_done__ :
          self.__ax__.cla()
          self.__canvas__.draw()
          self.__plot_done__ = False

        if self.__inputfile__ != "":

          self.__options_name_dialog__.setWindowTitle("Matrices name")
          self.__options_name_dialog__.exec_()
          
          try:
              msd = None
              bpd = None

              if str(self.__inputfile__).endswith('.csv'):
                  msd = basicutils.csvfile_to_mats(\
                          str(self.__inputfile__))
                  if msd == None:
                      QtWidgets.QMessageBox.critical( self, \
                              "ERROR", \
                              "Error while reading file")
                      return
                  bpd = basicutils.csvfile_to_mats(\
                          str(self.__inputfile__))
                  if bpd == None:
                      QtWidgets.QMessageBox.critical( self, \
                              "ERROR", \
                              "Error while reading file")
                      return
              elif str(self.__inputfile__).endswith('.mat'):
                  msd = scipy.io.loadmat(str(self.__inputfile__))
                  bpd = scipy.io.loadmat(str(self.__inputfile__))
              else:
                  QtWidgets.QMessageBox.critical( self, \
                          "ERROR", \
                          "Error in file extension " + self.__inputfile__ )
                  return

          except ValueError:
             QtWidgets.QMessageBox.critical( self, \
              "ERROR", \
              "Error while opening " + self.__inputfile__ )
             return
          
          if not(self.__options_name_dialog__.getratingname() in list(msd.keys())):
             QtWidgets.QMessageBox.critical( self, \
              "ERROR", \
              "Cannot find " + self.__options_name_dialog__.getratingname()+ \
              " in " + self.__inputfile__)
             return 
          
          if not(self.__options_name_dialog__.getiratingname() in list(bpd.keys())):
              QtWidgets.QMessageBox.critical( self, \
                      "ERROR", \
                      "Cannot find " + self.__options_name_dialog__.getiratingname() \
                      + " in " + self.__inputfile__)
              return 
          
          if msd[self.__options_name_dialog__.getratingname()].shape[0] != \
                  bpd[self.__options_name_dialog__.getiratingname()].shape[0]:
              QtWidgets.QMessageBox.critical( self, \
                      "ERROR", \
                      "wrong dim of the input matrix")
              return 

          self.__rm__ = msd[self.__options_name_dialog__.getratingname()].astype(numpy.int)
          self.__ir__ = bpd[self.__options_name_dialog__.getiratingname()].astype(numpy.float)
          self.__fileio__ = True

    def runchangepoint(self):

        if (self.__fileio__):
            self.__options_dialog_cp__.setWindowTitle("Options")
            self.__options_dialog_cp__.exec_()

           
            cp_fortest = -1
            num_of_run = 0
            cp_fortest_2 = -1
            cp_fortest_3 = -1

            #print self.__options_dialog_cp__.get_numofcp()
            #print self.__options_dialog_cp__.get_performtest()

            if len(self.__options_dialog_cp__.get_performtest()) < 1:
                QtWidgets.QMessageBox.critical( self, \
                        "ERROR", \
                        "Error in perform-test values")
                return

            cp_fortest = int(self.__options_dialog_cp__.get_performtest()[0])
            if cp_fortest > 0:
            
               if self.__options_dialog_cp__.get_numofcp() == 1:
                   if len(self.__options_dialog_cp__.get_performtest()) != 2:
                       QtWidgets.QMessageBox.critical( self, \
                               "ERROR", \
                               "Error in perform-test values")
                       return
                   
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[1])
               elif self.__options_dialog_cp__.get_numofcp() == 2:
                   if len(self.__options_dialog_cp__.get_performtest()) != 3:
                       QtWidgets.QMessageBox.critical( self, \
                               "ERROR", "Error in perform-test values")
                       return
                   
                   cp_fortest_2 = int(self.__options_dialog_cp__.get_performtest()[1])
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[2])
               elif self.__options_dialog_cp__.get_numofcp() == 3:
                   if len(self.__options_dialog_cp__.get_performtest()) != 4:
                       QtWidgets.QMessageBox.critical( self, \
                               "ERROR", \
                               "Error in perform-test values")
                       return
                   
                   cp_fortest_2 = int(self.__options_dialog_cp__.get_performtest()[1])
                   cp_fortest_3 = int(self.__options_dialog_cp__.get_performtest()[2])
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[3])

            progdialog = QtWidgets.QProgressDialog(
                    "", "Cancel", 0, 100.0, self)

            progdialog.setWindowTitle("Running")
            progdialog.setWindowModality(QtCore.Qt.WindowModal)
            progdialog.setMinimumDuration(0)
            progdialog.show()
            
            runcps = mkvtmod.changepoint()

            try:
                #print self.__options_dialog_cp__.get_cp1start(), " ", \
                #        self.__options_dialog_cp__.get_cp2start()," ", \
                #        self.__options_dialog_cp__.get_cp3start(), " ", \
                #        self.__options_dialog_cp__.get_cp1stop(), " ", \
                #        self.__options_dialog_cp__.get_cp2stop(), " ", \
                #        self.__options_dialog_cp__.get_deltacp()

                runcps.set_metacommunity (self.__rm__)
                runcps.set_num_of_bootstrap_iter (num_of_run)
                runcps.set_cp1_fortest (cp_fortest)
                runcps.set_cp2_fortest (cp_fortest_2)
                runcps.set_cp3_fortest (cp_fortest_3)

                runcps.set_num_of_cps (self.__options_dialog_cp__.get_numofcp())
                runcps.set_cp1_start_stop (\
                        self.__options_dialog_cp__.get_cp1start(),\
                        self.__options_dialog_cp__.get_cp1stop())
                runcps.set_cp2_start_stop (\
                        self.__options_dialog_cp__.get_cp2start(),\
                        self.__options_dialog_cp__.get_cp2stop())
                runcps.set_cp3_start_stop (\
                        self.__options_dialog_cp__.get_cp3start(),\
                        self.__options_dialog_cp__.get_cp3stop())
                runcps.set_delta_cp (\
                        self.__options_dialog_cp__.get_deltacp())
                runcps.set_print_iter_info(False)
                runcps.set_verbose(False)
               
                runcps.compute_cps (progdialog)

            except mkvtmod.Error:
                QtWidgets.QMessageBox.critical( self, \
                    "ERROR", \
                    "Oops! error in the main function")
                return
            except TypeError as err:
                QtWidgets.QMessageBox.critical( self, \
                    "ERROR", \
                    "Oops! type error in the main function")
                return
 

            if cp_fortest >= 0:
                QtWidgets.QMessageBox.information( self, \
                        "Value", "Lambda(95%) : " +\
                        str(runcps.get_lambda95()) + " Lambda : " + \
                        str(runcps.get_lambdastart()) + " P-Value : "+
                        str(runcps.get_pvalue()))
            else:
                if self.__options_dialog_cp__.get_numofcp() == 1:
                    QtWidgets.QMessageBox.information( self, \
                        "Value", "CP : " +\
                        str(runcps.get_cp1_found()) + " Value : " + \
                        str(runcps.get_maxval()))

                    x = []
                    y = []
                    for v in runcps.get_allvalues():
                        x.append(v[0])
                        y.append(v[1])

                    if self.__plot_done__ :
                      self.__ax__.cla()
                      self.__canvas__.draw()
                      self.__plot_done__ = False
                    else:
                      self.__ax__ = self.__figure__.add_subplot(111)
                    
                    self.__ax__.plot(x, y, '*-')
                    #self.__ax__.scatter(x, y)
                    self.__ax__.set_xlabel('Time')
                    self.__ax__.set_ylabel('Value')
                    self.__ax__.set_xlim([numpy.min(x), numpy.max(x)])
                    self.__ax__.annotate("ChangePoint", \
                            xy=(runcps.get_cp1_found(), \
                            runcps.get_maxval()), xytext=(-20, 20), \
                            textcoords='offset points', ha='right', va='bottom', \
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), \
                            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
                    self.__canvas__.draw()
                    self.__plot_done__ = True

                elif self.__options_dialog_cp__.get_numofcp() == 2:
                    QtWidgets.QMessageBox.information( self, \
                        "Value", "CP1 : " + str(runcps.get_cp1_found()) + \
                        " CP2 : " + str(runcps.get_cp2_found()) + \
                        " Value : " + str(runcps.get_maxval()))

                    if self.__plot_done__ :
                      self.__ax__.cla()
                      self.__canvas__.draw()
                      self.__plot_done__ = False

                elif self.__options_dialog_cp__.get_numofcp() == 3:
                    QtWidgets.QMessageBox.information( self, \
                        "Value", "CP1 : " + str(runcps.get_cp1_found()) + \
                        " CP2 : " + str(runcps.get_cp2_found()) + \
                        " CP3 : " + str(runcps.get_cp3_found()) + \
                        " Value : " + str(runcps.get_maxval()))

                    if self.__plot_done__ :
                      self.__ax__.cla()
                      self.__canvas__.draw()
                      self.__plot_done__ = False
 

            progdialog.setValue(100.0)
            progdialog.close()


        return

    def mainrun(self):

        self.__entropiadone__ = False
        self.__savefile__.setEnabled(False)
        self.__mats__.setEnabled(False)
        self.__plots__.setEnabled(False)

        if (self.__fileio__):

            if self.__plot_done__ :
                self.__ax__.cla()
                self.__canvas__.draw()
                self.__plot_done__ = False

            self.__options_dialog__.setWindowTitle("Options")

            self.__options_dialog__.exec_()

            progdialog = QtWidgets.QProgressDialog(
                    "", "Cancel", 0, 100.0, self)

            progdialog.setWindowTitle("Running")
            progdialog.setWindowModality(QtCore.Qt.WindowModal)
            progdialog.setMinimumDuration(0)
            progdialog.show()
            
            markovrun = mkvtmod.markovkernel()

            try:
            
                markovrun.set_metacommunity(self.__rm__)
                markovrun.set_attributes(self.__ir__)
                markovrun.set_step( \
                        self.__options_dialog__.getstep())
            
                markovrun.set_infinite_time( \
                        self.__options_dialog__.getinftime())
                markovrun.set_simulated_time( \
                        self.__options_dialog__.gettprev())
            
                markovrun.set_num_of_mc_iterations( \
                        self.__options_dialog__.getnofrun())
                markovrun.set_use_a_seed(False)
                markovrun.set_verbose(False)
                markovrun.set_dump_files(False)

                markovrun.set_usecopula(\
                        self.__options_dialog__.getusecopula())

                if not markovrun.run_computation(progdialog):
                    
                    QtWidgets.QMessageBox.critical( self, \
                    "ERROR", \
                    "Error in main markov kernel")

                    progdialog.setValue(100.0)
                    progdialog.close()
                
                    return 

            except TypeError as err:
                QtWidgets.QMessageBox.critical( self, \
                        "ERROR", err)

                progdialog.setValue(100.0)
                progdialog.close()
                
                return 

            self.__entropy__ = markovrun.get_entropy()
            self.__intra_entropy__ = markovrun.get_intra_entropy()
            self.__inter_entropy__ = markovrun.get_inter_entropy()
            self.__var__ = markovrun.get_entropy_sigma()
            self.__allratings__ = markovrun.get_attributes_pdf_values()
            self.__allratingsnins__ = markovrun.get_attributes_pdf_bins()
            self.__meanval__ = markovrun.get_attributes_mean_values()
            self.__stdeval__ = markovrun.get_attributes_sigma_values()
            self.__pr__ = markovrun.get_transitions_probability_mtx()

            progdialog.setValue(100.0)
            progdialog.close()

            self.__entropiadone__ = True
            self.__savefile__.setEnabled(True)
            self.__plots__.setEnabled(True)
            self.__mats__.setEnabled(True)

            if self.__options_dialog__.getinftime():
                ev = self.__entropy__[1:]
                QtWidgets.QMessageBox.information( self, \
                        "Value", "Stationary value: " +\
                        str(numpy.mean(ev)) + " stdev: " + \
                        str(numpy.std(ev)))
            else:
                self.plot()

        else:
            QtWidgets.QMessageBox.critical( self, \
                    "ERROR", \
                    "Error occurs while opening input file")

    def plot(self):

        x = []
        y = []
        intra_y = []
        inter_y = []
        for i in range(1,self.__options_dialog__.gettprev()+1):
            x.append(i)
            y.append(self.__entropy__[i-1])

            if self.__options_dialog__.getusecopula():
                intra_y.append(self.__intra_entropy__[i-1])
                inter_y.append(self.__inter_entropy__[i-1])
        
        if self.__entropiadone__ :
            if not self.__plot_done__ :
                self.__ax__  = self.__figure__.add_subplot(111)

            #self.__ax__.hold(False)
            self.__ax__.plot(x, y, '*-', label="DT")
            if self.__options_dialog__.getusecopula():
                self.__ax__.plot(x, intra_y, '*', label="Intra DT")
                self.__ax__.plot(x, inter_y, '-', label="Inter DT")
            #self.__ax__.scatter(x, y)
            self.__ax__.legend()
            self.__ax__.set_xlabel('Time')
            self.__ax__.set_ylabel('DT')
            self.__ax__.set_xlim([2, self.__options_dialog__.gettprev()])
            self.__canvas__.draw()
            self.__plot_done__ = True

    def view_mats(self):

        if self.__entropiadone__ :
            mw = matwin.matable(self.__pr__, "Transition matrix", self)

            mw.show()
  
        return 

    def plot_hist (self):

        if self.__entropiadone__ :
            ploh = plothist.plohitwin(self.__allratings__, \
                    self.__allratingsnins__, self)
            ploh.show()

    def get_options (self):

        return self.__options_dialog__


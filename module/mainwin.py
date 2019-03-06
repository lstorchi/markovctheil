from PyQt4 import QtGui, QtCore 

import mainmkvcmp
import changemod
import plothist
import scipy.io
import options
import matwin
import numpy
import os

import os.path

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class main_window(QtGui.QMainWindow):
    
    def __init__(self):
        self.__inputfile__ = ""
        self.__fileio__ = False
        self.__entropiadone__ = False
        self.__plot_done__ = False

        self.__usecopula__ = False
       
        QtGui.QMainWindow.__init__(self) 
        self.resize(640, 480) 
        self.setWindowTitle('Dynamic Theil entropy')
        self.statusBar().showMessage('Markov started') 

        ofile = QtGui.QAction(QtGui.QIcon("icons/open.png"), "Open", self)
        ofile.setShortcut("Ctrl+O")
        ofile.setStatusTip("Open file")
        ofile.connect(ofile, QtCore.SIGNAL('triggered()'), self.openfile)

        self.__savefile__ = QtGui.QAction(QtGui.QIcon("icons/save.png"), "Save", self)
        self.__savefile__.setStatusTip("Save file")
        self.__savefile__.connect(self.__savefile__ , QtCore.SIGNAL('triggered()'), \
                self.savefile)
        self.__savefile__.setEnabled(False)

        sep = QtGui.QAction(self)
        sep.setSeparator(True)

        quit = QtGui.QAction(QtGui.QIcon("icons/cancel.png"), "Quit", self)
        quit.setShortcut("Ctrl+Q")
        quit.setStatusTip("Quit application")
        self.connect(quit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))

        run = QtGui.QAction(QtGui.QIcon("icons/run.png"), "Run", self)
        run.setShortcut("Ctrl+R")
        run.setStatusTip("Run")
        self.connect(run, QtCore.SIGNAL('triggered()'), self.mainrun)

        runchangepoint = QtGui.QAction(QtGui.QIcon("icons/run.png"), "RunChangePoint", self)
        runchangepoint.setStatusTip("Run ChangePoint")
        self.connect(runchangepoint, QtCore.SIGNAL('triggered()'), self.runchangepoint)

        self.__plots__ = QtGui.QAction(QtGui.QIcon("icons/save.png"), "Plot CS distributions", self)
        self.__plots__.setShortcut("Ctrl+S")
        self.__plots__.setStatusTip("Credit spread distributions")
        self.__plots__.connect(self.__plots__ , QtCore.SIGNAL('triggered()'), \
                self.plot_hist)
        self.__plots__.setEnabled(False)

        self.__mats__ = QtGui.QAction(QtGui.QIcon("icons/save.png"), "View transition matrix", self)
        self.__mats__.setShortcut("Ctrl+M")
        self.__mats__.setStatusTip("View transition matrices")
        self.__mats__.connect(self.__mats__ , QtCore.SIGNAL('triggered()'), \
                self.view_mats)
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

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.__toolbar__)
        layout.addWidget(self.__canvas__)

        maindialog = QtGui.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)

        self.__options_dialog__ = options.optiondialog(self)

        self.__options_dialog_cp__ = options.optiondialog_cp(self)

        self.__options_name_dialog__ = options.optionnamedialog(self)

    def savefile(self):

        if self.__entropiadone__ :
           tosave = QtGui.QFileDialog.getSaveFileName(self) 
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
                ev = self.__entropia__[1:]
                m = numpy.mean(ev)
                s = numpy.std(ev)
                outf.write("Stationary value: " + \
                        str(m) + " stdev: " + str(s))
           else:
                outf.write("t E(DT) STD(DT)\n")
                for t in range(self.__options_dialog__.gettprev()):
                    outf.write("%d %f %f \n"%(t+1, self.__entropia__[t], \
                            self.__var__[t]))

           outf.close()

    def openfile(self):

        self.__inputfile__ = QtGui.QFileDialog.getOpenFileName(self) 

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
              msd = scipy.io.loadmat(str(self.__inputfile__))
              bpd = scipy.io.loadmat(str(self.__inputfile__))
          except ValueError:
             QtGui.QMessageBox.critical( self, \
              "ERROR", \
              "Error while opening " + self.__inputfile__ )
             return
          
          if not(self.__options_name_dialog__.getratingname() in msd.keys()):
             QtGui.QMessageBox.critical( self, \
              "ERROR", \
              "Cannot find " + self.__options_name_dialog__.getratingname()+ \
              " in " + self.__inputfile__)
             return 
          
          if not(self.__options_name_dialog__.getiratingname() in bpd.keys()):
              QtGui.QMessageBox.critical( self, \
                      "ERROR", \
                      "Cannot find " + self.__options_name_dialog__.getiratingname() \
                      + " in " + self.__inputfile__)
              return 
          
          if msd[self.__options_name_dialog__.getratingname()].shape[0] != \
                  bpd[self.__options_name_dialog__.getiratingname()].shape[0]:
              QtGui.QMessageBox.critical( self, \
                      "ERROR", \
                      "wrong dim of the input matrix")
              return 
          
          self.__rm__ = msd[self.__options_name_dialog__.getratingname()]
          self.__ir__ = bpd[self.__options_name_dialog__.getiratingname()]
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
                QtGui.QMessageBox.critical( self, \
                        "ERROR", \
                        "Error in perform-test values")
                return

            cp_fortest = int(self.__options_dialog_cp__.get_performtest()[0])
            if cp_fortest > 0:
            
               if self.__options_dialog_cp__.get_numofcp() == 1:
                   if len(self.__options_dialog_cp__.get_performtest()) != 2:
                       QtGui.QMessageBox.critical( self, \
                               "ERROR", \
                               "Error in perform-test values")
                       return
                   
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[1])
               elif self.__options_dialog_cp__.get_numofcp() == 2:
                   if len(self.__options_dialog_cp__.get_performtest()) != 3:
                       QtGui.QMessageBox.critical( self, \
                               "ERROR", "Error in perform-test values")
                       return
                   
                   cp_fortest_2 = int(self.__options_dialog_cp__.get_performtest()[1])
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[2])
               elif self.__options_dialog_cp__.get_numofcp() == 3:
                   if len(self.__options_dialog_cp__.get_performtest()) != 4:
                       QtGui.QMessageBox.critical( self, \
                               "ERROR", \
                               "Error in perform-test values")
                       return
                   
                   cp_fortest_2 = int(self.__options_dialog_cp__.get_performtest()[1])
                   cp_fortest_3 = int(self.__options_dialog_cp__.get_performtest()[2])
                   num_of_run = int(self.__options_dialog_cp__.get_performtest()[3])

            progdialog = QtGui.QProgressDialog(
                    "", "Cancel", 0, 100.0, self)

            progdialog.setWindowTitle("Running")
            progdialog.setWindowModality(QtCore.Qt.WindowModal)
            progdialog.setMinimumDuration(0)
            progdialog.show()

            try:
                #print self.__options_dialog_cp__.get_cp1start(), " ", \
                #        self.__options_dialog_cp__.get_cp2start()," ", \
                #        self.__options_dialog_cp__.get_cp3start(), " ", \
                #        self.__options_dialog_cp__.get_cp1stop(), " ", \
                #        self.__options_dialog_cp__.get_cp2stop(), " ", \
                #        self.__options_dialog_cp__.get_deltacp()

                vals = changemod.main_compute_cps (self.__rm__, \
                        num_of_run, cp_fortest, cp_fortest_2, cp_fortest_3, \
                        self.__options_dialog_cp__.get_numofcp(), \
                        self.__options_dialog_cp__.get_cp1start(), \
                        self.__options_dialog_cp__.get_cp2start(), \
                        self.__options_dialog_cp__.get_cp3start(), \
                        self.__options_dialog_cp__.get_cp1stop(), \
                        self.__options_dialog_cp__.get_cp2stop(), \
                        self.__options_dialog_cp__.get_cp3stop(), \
                        self.__options_dialog_cp__.get_deltacp(), \
                        False, None, False, progdialog)
                        #True, None, True, progdialog)
            except changemod.Error:
                QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "Oops! error in the main function")
                return

            if cp_fortest >= 0:
                assert(len(vals) == 3)
                QtGui.QMessageBox.information( self, \
                        "Value", "Lambda(95%) : " +\
                        str(vals[0]) + " Lambda : " + \
                        str(vals[1]) + " P-Value : "+
                        str(vals[2]))
            else:
                if self.__options_dialog_cp__.get_numofcp() == 1:
                    assert(len(vals) == 3)
                    QtGui.QMessageBox.information( self, \
                        "Value", "CP : " +\
                        str(vals[0]) + " Value : " + \
                        str(vals[1]))

                    x = []
                    y = []
                    for v in vals[2]:
                        x.append(v[0])
                        y.append(v[1])

                    if self.__plot_done__ :
                      self.__ax__.cla()
                      self.__canvas__.draw()
                      self.__plot_done__ = False
                    
                    self.__ax__ = self.__figure__.add_subplot(111)
                    self.__ax__.plot(x, y, '*-')
                    #self.__ax__.scatter(x, y)
                    self.__ax__.set_xlabel('Time')
                    self.__ax__.set_ylabel('Value')
                    self.__ax__.set_xlim([numpy.min(x), numpy.max(x)])
                    self.__ax__.annotate("ChangePoint", \
                            xy=(vals[0], vals[1]), xytext=(-20, 20), \
                            textcoords='offset points', ha='right', va='bottom', \
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), \
                            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
                    self.__canvas__.draw()
                    self.__plot_done__ = True

                elif self.__options_dialog_cp__.get_numofcp() == 2:
                    assert(len(vals) == 4)
                    QtGui.QMessageBox.information( self, \
                        "Value", "CP1 : " + str(vals[0]) + \
                        " CP2 : " + str(vals[1]) + \
                        " Value : " + str(vals[2]))

                    if self.__plot_done__ :
                      self.__ax__.cla()
                      self.__canvas__.draw()
                      self.__plot_done__ = False

                elif self.__options_dialog_cp__.get_numofcp() == 3:
                    assert(len(vals) == 5)
                    QtGui.QMessageBox.information( self, \
                        "Value", "CP1 : " + str(vals[0]) + \
                        " CP2 : " + str(vals[1]) + \
                        " CP3 : " + str(vals[2]) + \
                        " Value : " + str(vals[3]))

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

            progdialog = QtGui.QProgressDialog(
                    "", "Cancel", 0, 100.0, self)

            progdialog.setWindowTitle("Running")
            progdialog.setWindowModality(QtCore.Qt.WindowModal)
            progdialog.setMinimumDuration(0)
            progdialog.show()
            
            markovrun = mainmkvcmp.markovkernel()

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

                markovrun.set_usecopula(self.__usecopula__)

                if not markovrun.main_mkc_comp(progdialog):
                    
                    QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "Error in main markov kernel")

                    progdialog.setValue(100.0)
                    progdialog.close()
                
                    return 

            except TypeError as err:
                QtGui.QMessageBox.critical( self, \
                        "ERROR", err)

                progdialog.setValue(100.0)
                progdialog.close()
                
                return 

            self.__entropy__ = markovrun.get_entropy()
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
                ev = self.__entropia__[1:]
                QtGui.QMessageBox.information( self, \
                        "Value", "Stationary value: " +\
                        str(numpy.mean(ev)) + " stdev: " + \
                        str(numpy.std(ev)))
            else:
                self.plot(self.__entropia__)

        else:
            QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "Error occurs while opening input file")

    def plot(self, data):

        x = []
        y = []
        for i in range(1,self.__options_dialog__.gettprev()+1):
            x.append(i)
            y.append(data[i-1])
        
        if self.__entropiadone__ :
            self.__ax__  = self.__figure__.add_subplot(111)
            #self.__ax__.hold(False)
            self.__ax__.plot(x, y, '*-')
            #self.__ax__.scatter(x, y)
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


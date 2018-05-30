import time
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from twisted.internet.defer import inlineCallbacks

class MainWindow(QtGui.QMainWindow):
    def __init__(self, reactor, parent=None):
        super(MainWindow, self).__init__(parent)
        self.reactor = reactor
        self.setGeometry(200, 200, 1000, 450)
        self.setWindowTitle("NHMFL monitor")
        self.fig = pg.PlotWidget()
        self.setCentralWidget(self.fig)
        self.magconnect()
        self.t0 = time.time()
        self.create_graph()

        self.timer = QtCore.QTimer(self)  # initializing the Qtimer
        self.timer.timeout.connect(self.update)  # when the timer 'times out' call updateField
        self.timer.start(5000)  # starts the timer at 500 ms

    @inlineCallbacks
    def magconnect(self):
        """

        :return:
        """
        from labrad.wrappers import connectAsync
        cxn = yield connectAsync(name='MagoMon')
        self.mag = cxn.nhmfl_rmag
        self.mag.select_device()


    def create_graph(self):
        self.bfield= []
        self.plot1 = self.fig.plot()

    @inlineCallbacks
    def update(self):
        b = yield self.mag.status()[1]
        if len(self.bfield)>=18000:
            self.bfield = []
            self.bfield.append(float(b))
            pen1 = pg.mkPen('y', width=1.5)
            p1 = self.fig.plot(self.t, self.probe)
            p1.setPen(pen1)

    def closeEvent(self, e):
        self.reactor.stop()
        print "stop"

class Sweep1DWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Sweep1DWidget, self).__init__(parent)
        self.win1 = pg.GraphicsLayoutWidget



if __name__=="__main__":
    a = QtGui.QApplication( [] )
    import qt4reactor
    qt4reactor.install()
    from twisted.internet import reactor
    window = MainWindow(reactor)
    window.show()
    reactor.run()

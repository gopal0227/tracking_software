#! /usr/bin/env python
# version of tracker with pyqtgraph and cv
# mark two camera views at once,
# and show the 3d simultaneously

# update with drag pt
# change l and r to camera indexes

import numpy as n
import cv2 as cv
import os
import sys
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from scipy.interpolate import InterpolatedUnivariateSpline

qt_app = QtGui.QApplication(sys.argv)

class Stereography_window(QtGui.QMainWindow): #QWidget
    '''Window with two camera views, top, camera placement and 3d render, bottom.'''

    def __init__ (self):
        super(Stereography_window, self).__init__()
        self.setWindowTitle('Tracker 5')
        self.setMinimumWidth(1600)
        self.setMinimumHeight(1000)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.cur = QtGui.QCursor

        # Create the QVBoxLayout that lays out the whole form
        w = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(w)
        self.setCentralWidget(w)

        ### top row, imageview widgets
        self.images_hbox = QtGui.QHBoxLayout()
        self.iml = pg.ImageView()
        self.imr = pg.ImageView()
        # self.iml.setMouseTracking(True)
        # self.imr.setMouseTracking(True)
        # self.imr_load = QtGui.QPushButton('Load right directory')
        self.iml_load = QtGui.QPushButton('Load left Directory')
        self.lfns, self.rfns = [], []

        self.randim = n.random.randint(256, size=(640, 480))
        self.iml.setImage(self.randim)
        self.iml.setMinimumHeight(400)
        self.iml.getHistogramWidget().setMaximumWidth(100)
        self.iml.scene.sigMouseMoved.connect(self.lmouseMoved)
        self.imr.setImage(self.randim)
        self.imr.setMinimumHeight(400)
        self.imr.getHistogramWidget().setMaximumWidth(100)
        self.imr.scene.sigMouseMoved.connect(self.rmouseMoved)

        self.images_hbox.addWidget(self.iml)
        self.ltools = QtGui.QToolBar(self.iml)
        self.ltools.setStyleSheet('QToolBar{spacing:0px;}')
        self.loadl = self.ltools.addAction(QtGui.QIcon(), 'load', self.load)
        self.ltools.show()

        self.images_hbox.addWidget(self.imr)
        self.rtools = QtGui.QToolBar(self.imr)
        self.rtools.setStyleSheet('QToolBar{spacing:0px;}')
        # self.loadr = self.rtools.addAction(QtGui.QIcon(), 'load', self.load_right)
        self.rtools.show()

        ### middle row
        # frame slider
        self.frame_hbox = QtGui.QHBoxLayout()
        self.frame_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(1)
        self.frame_slider.valueChanged.connect(self.change_frame)
        # self.frame_value = pg.ValueLabel()
        self.frame_value = QtGui.QLabel('1/1')
        self.frame_hbox.addWidget(self.frame_slider)
        self.frame_hbox.addWidget(self.frame_value)

        ### bottom row
        self.bottom_hbox = QtGui.QHBoxLayout()
        # camera layout
        self.camera_view = pg.PlotWidget()
        self.camera_view.setAspectLocked(True)
        self.camera_view.setMaximumWidth(400)
        self.camera_view.setMaximumHeight(400)
        self.tab = pg.TableWidget(editable=True, sortable=False)
        self.tab.setData([[-14.13,0,14.13,0.],
                          [-15,0,15,0.],
                          [53.5,41.41,53.5,41.41]])
        self.tab.setVerticalHeaderLabels(['pos', 'ang', 'fov'])
        self.tab.setHorizontalHeaderLabels(['cam1 x', 'cam1 y', 'cam2 x', 'cam2 y'])
        self.tab.cellChanged.connect(self.update_camera_parameters)
        self.update_camera_parameters(dist=100)
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.camera_view)
        vbox.addWidget(self.tab)
        self.bottom_hbox.addItem(vbox)

        # 3d plot
        self.flight_view = gl.GLViewWidget()
        self.flight_view.setBackgroundColor((0, 0, .5))
        self.flight_view.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.flight_view.opts['distance'] = 40
        gx = gl.GLGridItem(color=pg.mkColor([255,0,0,255])) 
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.flight_view.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.flight_view.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.flight_view.addItem(gz)
        # self.flight_view.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # self.flight_view.show()
        self.flight_lines = []
        self.flight_pts = []
        self.bottom_hbox.addWidget(self.flight_view)

        ### set the layout
        self.layout.addLayout(self.images_hbox)
        # self.layout.addWidget(self.frame_slider)
        self.layout.addLayout(self.frame_hbox)
        self.layout.addLayout(self.bottom_hbox)


        ### now the markers
        self.num_cams = 2    #left and right
        self.num_markers = 9  #1-9
        self.num_frames = 1   #until we load a dir
        self.marker_keys = [str(i+1) for i in range(self.num_markers)] # keyboard strings for each marker
        self.shift_marker_keys = ['!', '@', '#', '$', '%', '^', '&', '*', '('] # keyboard strings for each marker
        alpha = 200
        width = 4
        self.colors = [(255,100,100,alpha), #red
                       (100,255,100,alpha), #green
                       ( 60, 60,255,alpha), #blue
                       (245,245, 30,alpha), #yellow
                       ( 30,245,245,alpha), #cyan
                       (255,  0,255,alpha), #magenta
                       (255,195,  0,alpha), #orange
                       (150,150,255,alpha), #indigo
                       (215, 120,255,alpha)] #purple
        # first the markers for the image
        self.data_markers = [[], []]
        for cam_ind in range(self.num_cams):
            for marker_ind in range(self.num_markers):
                # left
                data_marker = pg.PolyLineROI([[0.,0.]]) #roi with only a single point
                data_marker.getHandles()[0].pen.setColor(QtGui.QColor(*self.colors[marker_ind]))#make each handle a different color
                data_marker.getHandles()[0].pen.setWidth(width)                   #thicken lines
                data_marker.sigRegionChanged.connect(self.marker_moved)
                data_marker.hide() #initially invisible
                self.data_markers[cam_ind].append(data_marker)
                if cam_ind == 0: self.iml.addItem(data_marker)
                if cam_ind == 1: self.imr.addItem(data_marker)
        self.data = n.zeros((self.num_cams + 1, self.num_markers, 3, self.num_frames))
        self.data[:,:,:,:] = n.NaN
        # data positions interpolated
        self.null_interp = InterpolatedUnivariateSpline([0,0], [n.NaN,n.NaN], k=1)
        # self.data_interp = [[[self.null_interp]*2]*self.num_markers]*self.num_sides #[2 sides][num markers][x y] ##no t needed
        self.data_interp = [[[self.null_interp for xy in range(2)] for m in range(self.num_markers)] for s in range(self.num_cams)]
        # now the lines
        for marker_ind in range(self.num_markers):
            line = gl.GLLinePlotItem(pos=n.array([[0,0,0.],[1,1,1.]]),
                                     color=pg.glColor(self.colors[marker_ind]), width=2., antialias=True)
            line.hide()
            self.flight_lines.append(line)
            self.flight_view.addItem(line)
            pt = gl.GLScatterPlotItem(pos=n.array([[0,0,0.]]),
                                     color=pg.glColor(self.colors[marker_ind]), size=10.)
            pt.hide()
            self.flight_pts.append(pt)
            self.flight_view.addItem(pt)
                                      

    def load(self, dirname=None, tryright=True):
        if dirname==None:
            dirname = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        if dirname.endswith('l'):
            ldirname = dirname
            rdirname = dirname[:-1] + 'r'
        elif dirname.endswith('r'):
            rdirname = dirname
            ldirname = dirname[:-1] + 'l'
        self.savename = dirname[:-1] + '_data.npy'
        self.lfns = [ldirname + '/' + f for f in os.listdir(ldirname) if f.endswith(('.png', 'PNG', '.jpg', 'JPG'))]
        self.lfns.sort()
        self.rfns = [rdirname + '/' + f for f in os.listdir(rdirname) if f.endswith(('.png', 'PNG', '.jpg', 'JPG'))]
        self.rfns.sort()
        if tryright:
            # self.load_right(dirname[:-1] + 'r')
            dirname = dirname[:-1] + 'r'
            ls = os.listdir(dirname)
            self.rfns = [dirname + '/' + f for f in ls if f.endswith(('.png', 'PNG', '.jpg', 'JPG'))]
            self.rfns.sort()
            self.frame_slider.setValue(0)
            self.num_frames = max(len(self.lfns), len(self.rfns))
            self.frame_slider.setMaximum(self.num_frames)
            self.change_frame()
            self.rres = self.imr.image.shape
        self.change_frame()
        self.lres = self.iml.image.shape
        if os.path.isfile(self.savename): #search up one dir for data file
            print 'loading'
            self.data = n.load(self.savename)
            self.num_frames = self.data.shape[-1]
            # make all the maker intepolations
            for cam_ind in range(self.num_cams):
                for marker_ind in range(self.num_markers):
                    self.make_interp(cam_ind, marker_ind)
        else:
            self.num_frames = max(len(self.lfns), len(self.rfns))
            self.data = n.zeros((self.num_cams + 1, self.num_markers, 3, self.num_frames))
        self.frame_slider.setMaximum(self.num_frames)
        self.frame_slider.setValue(0)
        self.show_markers()
        self.camera_to_3d()
        self.draw_lines()
        self.setWindowTitle(self.savename)

        
    def change_frame(self):
        frame_ind = self.frame_slider.value() - 1
        self.frame_value.setText('{}/{}'.format(self.frame_slider.value(), self.num_frames))
        if frame_ind >= len(self.lfns):
            self.iml.setImage(self.randim, autoRange=False, autoLevels=False)
        else:
            self.iml.setImage(cv.imread(self.lfns[frame_ind], 0).T, autoRange=False, autoLevels=False)
        if frame_ind >= len(self.rfns):
            self.imr.setImage(self.randim, autoRange=False, autoLevels=False)
        else:
            self.imr.setImage(cv.imread(self.rfns[frame_ind], 0).T, autoRange=False, autoLevels=False)
        self.show_markers()
        self.draw_pts()

    def move_frame_slider(self, move=0):
        curr_value = self.frame_slider.value() - 1
        new_value = n.mod(curr_value + move, self.frame_slider.maximum())
        self.frame_slider.setValue(new_value + 1)
        self.change_frame()
        
    def show_markers(self):
        frame_ind = self.frame_slider.value() - 1
        for cam_ind in range(self.num_cams):
            for marker_ind in range(self.num_markers):
                if self.data[cam_ind, marker_ind, -1, frame_ind]==1:
                    x, y = self.data[cam_ind, marker_ind, :2, frame_ind]
                    width = 4
                else:
                    x = self.data_interp[cam_ind][marker_ind][0](frame_ind)
                    y = self.data_interp[cam_ind][marker_ind][1](frame_ind)
                    width = 1
                if n.isnan(x+y):
                    self.data_markers[cam_ind][marker_ind].hide()
                else:
                    self.data_markers[cam_ind][marker_ind].show()
                    self.data_markers[cam_ind][marker_ind].setPos((x,y))
                    self.data_markers[cam_ind][marker_ind].getHandles()[0].pen.setWidth(width)
                    

    def update_camera_parameters(self, row=0, col=0, dist=100):
        self.camera_view.clear()
        cam1xpos, cam1ypos, cam2xpos, cam2ypos = self.tab.item(0,0).value, self.tab.item(0,1).value, self.tab.item(0,2).value, self.tab.item(0,3).value
        cam1xang, cam1yang, cam2xang, cam2yang = self.tab.item(1,0).value, self.tab.item(1,1).value, self.tab.item(1,2).value, self.tab.item(1,3).value
        cam1xfov, cam1yfov, cam2xfov, cam2yfov = self.tab.item(2,0).value, self.tab.item(2,1).value, self.tab.item(2,2).value, self.tab.item(2,3).value
        # 1
        alpha = cam1xfov/2.*n.pi/180.
        dev = dist*n.tan(alpha)
        points = n.array([[-dev,0.,-5,-5,5,5,0,dev],[dist,0.,0,-2,-2,0,0,dist]])
        t = cam1xang*n.pi/180.
        rotmat = n.array([[n.cos(t), -n.sin(t)], [n.sin(t), n.cos(t)]])
        points = n.dot(rotmat, points)
        translate = n.array([cam1xpos, cam1ypos])
        points = points + translate[:,n.newaxis]
        self.camera_view.plot(points[0], points[1], symbol=None)
        # 2
        alpha = cam2xfov/2.*n.pi/180.
        dev = dist*n.tan(alpha)
        points = n.array([[-dev,0.,-5,-5,5,5,0,dev],[dist,0.,0,-2,-2,0,0,dist]])
        t = cam2xang*n.pi/180.
        rotmat = n.array([[n.cos(t), -n.sin(t)], [n.sin(t), n.cos(t)]])
        points = n.dot(rotmat, points)
        translate = n.array([cam2xpos, cam2ypos])
        points = points + translate[:,n.newaxis]
        self.camera_view.plot(points[0], points[1], symbol=None)

    def keyPressEvent(self, ev):
        # self.ev = ev
        if ev.key()<256:
            if chr(ev.key()) in self.marker_keys:
                self.move_marker(ev)
            elif chr(ev.key()) in self.shift_marker_keys:
                self.remove_marker(ev)
            elif chr(ev.key()) == '=':
                self.move_frame_slider(1)
            elif chr(ev.key()) == '-':
                self.move_frame_slider(-1)
            elif chr(ev.key()) == '+':
                self.move_frame_slider(10)
            elif chr(ev.key()) == '_':
                self.move_frame_slider(-10)
            # print chr (ev.key())


    def move_marker(self, ev=None):
        if chr(ev.key()) in self.marker_keys:
            marker_ind = self.marker_keys.index(chr(ev.key()))
            frame_ind = self.frame_slider.value() - 1
            if self.iml.imageItem.isUnderMouse(): # check left image
                x = self.iml.getImageItem().mapFromScene(self.imlpos).x()
                y = self.iml.getImageItem().mapFromScene(self.imlpos).y()
                self.data[0, marker_ind,:,frame_ind] = [x,y,1]
                self.make_interp(0, marker_ind)
            elif self.imr.imageItem.isUnderMouse(): # check right image
                x = self.imr.getImageItem().mapFromScene(self.imrpos).x()
                y = self.imr.getImageItem().mapFromScene(self.imrpos).y()
                self.data[1, marker_ind,:,frame_ind] = [x,y,1]
                self.make_interp(1, marker_ind)
            self.save()
            self.show_markers()
            self.camera_to_3d()
            self.draw_lines()


    def remove_marker(self, ev=None):
        if chr(ev.key()) in self.shift_marker_keys:
            marker_ind = self.shift_marker_keys.index(chr(ev.key()))
            frame_ind = self.frame_slider.value() - 1
            if self.iml.imageItem.isUnderMouse(): # check left image
                self.data[0, marker_ind,:,frame_ind] = [n.NaN, n.NaN, 0.]
                self.make_interp(0, marker_ind)
            elif self.imr.imageItem.isUnderMouse(): # check left image
                self.data[1, marker_ind,:,frame_ind] = [n.NaN, n.NaN, 0.]
                self.make_interp(1, marker_ind)
            self.show_markers()
            self.camera_to_3d()
            self.draw_lines()

    def marker_moved(self, pos=None):
        # print ('marker_moved', pos)
        pass

    def lmouseMoved(self, pos):
        self.imlpos = pos

    def rmouseMoved(self, pos):
        self.imrpos = pos

    def make_interp(self, cam_ind, marker_ind):
        # self.interp[lr][marker_ind] =
        num_filled_frames = sum(self.data[cam_ind, marker_ind, -1])
        if num_filled_frames>=2:
            kval = n.clip(num_filled_frames -1, 1,3)
            ts = n.arange(self.num_frames)[self.data[cam_ind, marker_ind, -1]==1]
            xs = self.data[cam_ind, marker_ind, 0][self.data[cam_ind, marker_ind, -1]==1]
            ys = self.data[cam_ind, marker_ind, 1][self.data[cam_ind, marker_ind, -1]==1]
            self.data_interp[cam_ind][marker_ind][0] = InterpolatedUnivariateSpline(ts, xs, k=kval)
            self.data_interp[cam_ind][marker_ind][1] = InterpolatedUnivariateSpline(ts, ys, k=kval)
        else:
            self.data_interp[cam_ind][marker_ind] = [self.null_interp, self.null_interp]

    def line_to_line(self, l1, l2):
        '''Returns the point midway between the closest approach of two lines.'''
        l1p0, l1p1 = l1[0], l1[1]
        l2p0, l2p1 = l2[0], l2[1]
        u, v, w = l1p1 - l1p0, l2p1 - l2p0, l1p0 - l2p0
        a, b, c, d, e = n.dot(u, u), n.dot(u, v), n.dot(v, v), n.dot(u, w), n.dot(v, w)
        D = a*c - b**2
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D
        l1sc = l1p0 + sc*(l1p1 - l1p0)
        l2tc = l2p0 + sc*(l2p1 - l2p0)
        midpoint = (l1sc + l2tc)/2.
        return midpoint

    # def pt_to_ln(self, pt, campos, camang, fov=[30, 24], res=[1280, 512]):
    def pt_to_ln(self, cam_ind, marker_ind, frame_ind):
        '''calculate the line specified by an x and y pt in an image and an initial
        camera position and angle.
        '''
        if cam_ind == 0:
            camxpos, camypos = self.tab.item(0,0).value, self.tab.item(0,1).value
            camxang, camyang = self.tab.item(1,0).value, self.tab.item(1,1).value
            camxfov, camyfov = self.tab.item(2,0).value, self.tab.item(2,1).value
            camzpos = 0.
        else:
            camxpos, camypos = self.tab.item(0,2).value, self.tab.item(0,3).value
            camxang, camyang = self.tab.item(1,2).value, self.tab.item(1,3).value
            camxfov, camyfov = self.tab.item(2,2).value, self.tab.item(2,3).value
            camzpos = 0.
            
        # get the width of each pixel in the horizontal and vertical directions
        ptx_width = float(camxfov)/self.lres[0]
        pty_width = float(camyfov)/self.lres[1]

        # the angles at the edge of the cameras field of view, pt = 0,0
        x0_ang = camxang + 90 + camxfov/2.
        y0_ang = camyang + camyfov/2.

        # calculate x and y angles relative to the camera represented by each pt
        mx = self.data_interp[cam_ind][marker_ind][0](frame_ind)
        my = self.data_interp[cam_ind][marker_ind][1](frame_ind)
        xang = x0_ang - mx * ptx_width
        yang = y0_ang - my * pty_width

        # get all the coordinates of a point on the projected line, a distance of 1 away    
        x = n.cos(xang*n.pi/180) + camxpos
        y = n.sin(yang*n.pi/180) + camypos
        z = (n.sin(xang*n.pi/180) + camzpos)*n.cos(yang*n.pi/180)

        # return a line, described as a list of 2 pts on it
        return n.array([[camxpos, camypos, camzpos], [x, z, y]])

    def camera_to_3d(self):
        for frame_ind in range(self.num_frames):
            for marker_ind in range(self.num_markers):
                l0 = self.pt_to_ln(0, marker_ind, frame_ind)
                l1 = self.pt_to_ln(1, marker_ind, frame_ind)
                self.data[-1, marker_ind, :, frame_ind] = self.line_to_line(l0, l1)

    def check_marked_overlap(self, marker_ind):
        if all(s.data[:-1,marker_ind,-1,:].sum(1)>2): #if each camera has at least a couple of marked frames
            cams, frames = n.where(self.data[:,marker_ind, -1,:])
            min_frame = n.max([n.min(frames[n.where(cams==cam)]) for cam in n.unique(cams)]) #the largest of all the first marked frames for each cam
            max_frame = n.min([n.max(frames[n.where(cams==cam)]) for cam in n.unique(cams)]) #the smallest of all the last marked frames for each cam
            if min_frame<max_frame:
                return min_frame, max_frame
            else:
                return None
        else:
            return None

    def draw_lines(self):
        minmaxlist = n.zeros((self.num_markers,2,3)) #marker, min or max, x y z
        overlaps = [self.check_marked_overlap(marker_ind) for marker_ind in range(self.num_markers)]
        for marker_ind in range(self.num_markers):
            overlap = overlaps[marker_ind]
            if overlap:
                minmaxlist[marker_ind, 0, :] = self.data[-1, marker_ind, :, overlap[0]:overlap[1]].min(1)
                minmaxlist[marker_ind, 1, :] = self.data[-1, marker_ind, :, overlap[0]:overlap[1]].max(1)
            else:
                minmaxlist[marker_ind, 0, :] = [n.inf, n.inf, n.inf]
                minmaxlist[marker_ind, 1, :] = [-n.inf, -n.inf, -n.inf]
        mins = minmaxlist[:,0].min(0)
        maxs = minmaxlist[:,1].max(0)
        ptps = maxs - mins
        centers = mins + ptps/2.
        maxscale = ptps.max()
        for marker_ind in range(self.num_markers):
            overlap = overlaps[marker_ind]
            if overlap:
                d = self.data[-1, marker_ind, :, overlap[0]:overlap[1]].T
                d -= centers
                d /= maxscale/20
                self.flight_lines[marker_ind].setData(pos=d)
                self.flight_lines[marker_ind].show()
            else:
                self.flight_lines[marker_ind].hide()
        self.draw_pts()

    def draw_pts(self):
        overlaps = [self.check_marked_overlap(marker_ind) for marker_ind in range(self.num_markers)]
        frame = self.frame_slider.value()
        for marker_ind in range(self.num_markers):
            overlap = overlaps[marker_ind]
            if overlap and overlap[0]<frame<overlap[1]:
                self.flight_pts[marker_ind].setData(pos=self.flight_lines[marker_ind].pos[frame-overlap[0]])
                self.flight_pts[marker_ind].show()
            else:
                self.flight_pts[marker_ind].hide()
                

    def save(self):
        # save cam
        # save data
        data_name = self.lfns[0].rsplit('/',2)[0] + '/data.npy'
        data_name = self.savename
        n.save(data_name, self.data)

    def run(self):
        self.show()
        pg.QtGui.QApplication.exec_()

        

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        # pg.QtGui.QApplication.exec_()
        s = Stereography_window()
        s.run()



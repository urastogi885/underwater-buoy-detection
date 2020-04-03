'''Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.

See the file example.py for an application. 

Created by Joerg Doepfert 2014 based on code posted by Daniel
Kornhauser.

'''


import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
from copy import deepcopy


class roipoly:

    def __init__(self, fig=[], ax=[], roicolor='b'):
        if fig == []:
            fig = plt.gcf()

        if ax == []:
            ax = plt.gca()

        self.previous_point = []
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.roicolor = roicolor
        self.fig = fig
        self.ax = ax
        self.buoy_points = None
        self.merge_list = []

        self.__ID1 = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.__motion_notify_callback)
        self.__ID2 = self.fig.canvas.mpl_connect(
            'button_press_event', self.__button_press_callback)

        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show()

    def getMask(self, currentImage):
        ny, nx = np.shape(currentImage)
        poly_verts = [(self.allxpoints[0], self.allypoints[0])]
        for i in range(len(self.allxpoints)-1, -1, -1):
            poly_verts.append((self.allxpoints[i], self.allypoints[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        ROIpath = mpl_path.Path(poly_verts)
        grid = ROIpath.contains_points(points).reshape((ny,nx))
        return grid
      
    def displayROI(self, **linekwargs):
        l = plt.Line2D(self.allxpoints +
                     [self.allxpoints[0]],
                     self.allypoints +
                     [self.allypoints[0]],
                     color=self.roicolor, **linekwargs)
        ax = plt.gca()
        ax.add_line(l)
        plt.draw()

    def display_mean(self, current_image, **text_kwargs):
        mask = self.getMask(current_image)
        mean_val = np.mean(np.extract(mask, current_image))
        std_val = np.std(np.extract(mask, current_image))
        string = "%.3f +- %.3f" % (mean_val, std_val)
        plt.text(self.allxpoints[0], self.allypoints[0],
                 string, color=self.roicolor,
                 bbox=dict(facecolor='w', alpha=0.6), **text_kwargs)

    def merge(self, list1, list2):
        """
        Add extraction point to a list
        :param list1: x-points on image
        :param list2: y-points on image
        :return: nothing
        """
        self.merge_list = [(list1[i], list2[i]) for i in range(0, len(list1))]

    def empty_list(self):
        """
        Remove all elements from merge-list
        :return: a list with all extraction points if condition is satisfied
        """
        # TODO: Add constants.py to define project constants
        if len(self.merge_list) == 8:
            self.buoy_points = deepcopy(self.merge_list)
            self.merge_list.clear()
            return self.buoy_points
        return None

    def __motion_notify_callback(self, event):
        if event.inaxes:
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if (event.button == None or event.button == 1) and self.line != None: # Move line around
                self.line.set_data([self.previous_point[0], x],
                                   [self.previous_point[1], y])
                self.fig.canvas.draw()


    def __button_press_callback(self, event):
        # buoy_pts = []
        if event.inaxes:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1 and event.dblclick == False:  # If you press the left button, single click
                if self.line == None: # if there is no line, create a line
                    self.line = plt.Line2D([x, x],
                                           [y, y],
                                           marker='o',
                                           color=self.roicolor)
                    self.start_point = [x, y]
                    print('x = %d, y= %d' % (x,y))
                    # buoy_pts.append((x,y))
                    self.previous_point = self.start_point
                    self.allxpoints=[x]
                    self.allypoints=[y]
                    ax.add_line(self.line)
                    self.fig.canvas.draw()

                    # add a segment
                else: # if there is a line, create a segment
                    self.line = plt.Line2D([self.previous_point[0], x],
                                           [self.previous_point[1], y],
                                           marker = 'o',color=self.roicolor)
                    self.previous_point = [x,y]
                    self.allxpoints.append(x)
                    self.allypoints.append(y)
                    print('x = %d, y= %d' % (x, y))
                    # buoy_pts.append((x,y))
                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()
                    # print(buoy_pts)
                    self.merge(self.allxpoints, self.allypoints)
                    self.empty_list()

            elif ((event.button == 1 and event.dblclick==True) or
                  (event.button == 3 and event.dblclick==False)) and self.line != None: # close the loop and disconnect
                self.fig.canvas.mpl_disconnect(self.__ID1) #joerg
                self.fig.canvas.mpl_disconnect(self.__ID2) #joerg
                        
                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None
                        
                if sys.flags.interactive:
                    pass
                else:
                    #figure has to be closed so that code can continue
                    plt.close(self.fig)
        # return all_pos
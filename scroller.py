# Inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/event_handling/image_slices_viewer.py

# Usage - if numpy file volume_i.npy exists:
# python scroller.py i

# Mass Usage - if numpy files volume_0.npy...volume_N.npy exist:
# for i in {0..N};  do python scroller.py $i; done;

'''
Abhi: for i in {0..1169};  do python scroller.py $i; done;
Alex: for i in {1170..2339};  do python scroller.py $i; done;
Vin:  for i in {2340..3512};  do python scroller.py $i; done;
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import sys
import glob
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons


# current volume number
volume_number = sys.argv[1]
print(volume_number)

class Labels():
    def __init__(self, volume_n):
        self.volume_n = volume_n
        self.up = False
        self.normal = False
        self.cancer = False
        self.brain_anomaly = False
        self.other_anomaly = False
        self.faulty = False

    def toggle_up(self, _):
        self.up = not self.up

    def toggle_normal(self, _):
        # print(a)
        self.normal = not self.normal
    def toggle_cancer(self, _):
        self.cancer = not self.cancer
    def toggle_ba(self, _):
        self.brain_anomaly = not self.brain_anomaly
    def toggle_oa(self, _):
        self.other_anomaly = not self.other_anomaly
    def toggle_faulty(self, _):
        self.faulty = not self.faulty

    def write_file(self, _):
        f = open('manual_volume_labels.txt', 'a+')

        f.write(str(volume_number))
        if self.up:
            f.write(",up")
        if self.normal:
            f.write(",normal")
        if self.cancer:
            f.write(",cancer")
        if self.brain_anomaly:
            f.write(",brain_anomaly")
        if self.other_anomaly:
            f.write(",other_anomaly")
        if self.faulty:
            f.write(",faulty")

        f.write("\n")

        f.close()
        sys.exit()




# handle scrolling through volume
class IndexTracker(object):
    def __init__(self, ax, X, n):
        self.ax = ax
        ax.set_title('scrolling through VOLUME {}'.format(n))
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=0, vmax=1)
        self.update()

    def onscroll(self, event):
        if event.button == 'up' and self.ind < self.slices - 1:
            self.ind = self.ind + 1
        elif event.button == 'down' and self.ind > 0:
            self.ind = self.ind - 1
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


# handle selecting volume category

# five buttons to label volume as belonging to five different categories
def on_click(event):
    pass

# checkbox to decide whether patient is looking up or to the front
def checkbox(label):
    # global up
    label = not(label)


# # append category and corresponding volume number to file
# def append_to_file(text):
#     f.write("{}\n".format(text))



# plot everything
fig = plt.figure(figsize=(8.5,5.5))
ax = plt.subplot2grid((1,1), (0,0),)

X = np.load('/home/abhishekmoturu/Desktop/gan_cancer_detection/brain_mri_512/volume_{}.npy'.format(volume_number)).astype(np.float32)

label = Labels(volume_number)

rax = plt.axes([0.05, 0.4, 0.1, 0.2])
up_check = CheckButtons(rax, ('UP',), (False,))
up_check.on_clicked(label.toggle_up)

rax = plt.axes([0.05, 0.2, 0.1, 0.2])
next = Button(rax, "NEXT", color='white', hovercolor='green')
next.on_clicked(label.write_file)


tracker = IndexTracker(ax, X, volume_number)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.connect('button_press_event', on_click)

axcut = plt.axes([0.83, 0.75, 0.15, 0.1])
n_cut = CheckButtons(axcut, ('NORMAL',), (False,))#, color='white', hovercolor='green')
n_cut.on_clicked(label.toggle_normal)

axcut = plt.axes([0.83, 0.6, 0.15, 0.1])
c_cut = CheckButtons(axcut, ('CANCER',), (False,))#Button(axcut, 'CANCER', color='white', hovercolor='yellow')
c_cut.on_clicked(label.toggle_cancer)

axcut = plt.axes([0.83, 0.45, 0.15, 0.1])
ba_cut = CheckButtons(axcut, ('BRAIN ANOMALY',), (False,))#Button(axcut, 'BRAIN ANOMALY', color='white', hovercolor='orange')
ba_cut.on_clicked(label.toggle_ba)

axcut = plt.axes([0.83, 0.30, 0.15, 0.1])
oa_cut = CheckButtons(axcut, ('OTHER ANOMALY',), (False,))#Button(axcut, 'OTHER ANOMALY', color='white', hovercolor='orange')
oa_cut.on_clicked(label.toggle_oa)

axcut = plt.axes([0.83, 0.15, 0.15, 0.1])
f_cut = CheckButtons(axcut, ('FAULTY',), (False,))#Button(axcut, 'FAULTY', color='white', hovercolor='red')
f_cut.on_clicked(label.toggle_faulty)

plt.show()


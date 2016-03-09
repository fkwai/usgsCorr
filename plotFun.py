import numpy as np
import matplotlib.pyplot as plt

def plot121line(ax):
    xmin,xmax=ax.get_xlim()
    ymin,ymax=ax.get_ylim()
    vmin=np.min([xmin,ymin])
    vmax=np.max([xmax,ymax])
    ax.set_xlim(vmin,vmax)
    ax.set_ylim(vmin,vmax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
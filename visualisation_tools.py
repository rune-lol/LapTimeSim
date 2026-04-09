import matplotlib.pyplot as plt
import math

def draw_track(track, values, cmap = 'plasma', valname='Value'):
    heading = [0]*len(track)
    prev_heading = 0
    for i, seg in enumerate(track):
        heading[i] = seg[1]*seg[0] + prev_heading
        prev_heading = heading[i]

    x = []
    y = []
    lastCoord = (0, 0)
    for i, _ in enumerate(track):
        x_n = lastCoord[0] + track[i][0]*math.cos(heading[i])
        y_n = lastCoord[1] + track[i][0]*math.sin(heading[i])
        x.append(x_n)
        y.append(y_n)
        new_coord = (x_n, y_n)
        lastCoord = new_coord

    plt.figure(figsize=(6, 3))
    plt.scatter(x, y, c=values, cmap=cmap)
    plt.colorbar(label=valname)
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track")
    plt.show()

def draw_distancetrace(track, list_values, valname='Value'):
    cum_distance = []
    prev_seg = 0
    for seg in track:
        cum_distance.append(seg[0]+prev_seg)
        prev_seg = seg[0]+prev_seg

    plt.figure(figsize=(6, 3))
    cmap = get_cmap(len(list_values)+1)

    fig, ax = plt.subplots()
    ax.plot(cum_distance, list_values[0][0], color=cmap(0))
    ax.set_ylabel(list_values[0][1], color=cmap(0))
    ax.tick_params(axis='y', labelcolor=cmap(0))
    
    for i, value in enumerate(list_values[1:]):
        ax2 = ax.twinx()
        ax2.plot(cum_distance, value[0], color=cmap(i+1))
        ax2.set_ylabel(value[1], color=cmap(i+1))
        ax2.tick_params(axis='y', labelcolor=cmap(i+1))
    plt.xlabel("x [m]")
    plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
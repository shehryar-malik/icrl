
import numpy as np
import matplotlib.pyplot as plt


### Check if two line segments intersect.
### https://kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python

def on_segment(p, q, r):
    if (r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and
        r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1])):
        return True
    return False

def orientation(p, q, r):
    val = (((q[1] - p[1]) * (r[0] - q[0])) -
           ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        return 0
    return 1 if val > 0 else -1

def intersects(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q1, q2): return True
    if o3 == 0 and on_segment(p2, q2, p1): return True
    if o4 == 0 and on_segment(p2, q2, q1): return True

    return False

###
def in_regions(prev_state, next_state, regions):
        """Returns True if agent moves through/on rectangles defined
        by `regions'."""
        for region in regions:
            if in_rectangle(prev_state, region):
                return True
            if in_rectangle(next_state, region):
                return True
            for bound in boundaries(*region):
                if intersects((prev_state, next_state), bound):
                    return True
        return False

def boundaries(o, w, h):
    """Returns the boundaries of rectangle of width w and height h with the
    bottom left corner at the point o.
    """
    return [(o, o + np.array([w,0])),
            (o, o + np.array([0,h])),
            (o + np.array([w,0]), o + np.array([w, h])),
            (o + np.array([0,h]), o + np.array([w, h]))]

def in_rectangle(state, region):
    """Returns True if a state ((x,y) coordinate) is in a rectangle defined
    by region (a tuple of origin, width and height).
    """
    o, w, h = region
    if (state[0] > o[0] and state[0] < o[0] + w and
        state[1] > o[1] and state[1] < o[1] + h):
        return True

def add_circle(ax, point, color, radius=0.2, clip_on=False):
    circle = plt.Circle(
            point,
            radius=radius,
            color=color,
            clip_on=clip_on
    )
    ax.add_patch(circle)

def figure_to_array(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image



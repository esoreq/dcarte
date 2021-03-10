import numpy as np
import scipy.interpolate as si
import bz2
from functools import wraps


def timer(text=None, pre_text=None, post_text=None):
    def wrapper(f):
        @wraps(f)
        def wrapped(other, *f_args, **f_kwargs):
            if other.verbose:
                _start = time.perf_counter()
            out = f(other, *f_args, **f_kwargs)
            if other.verbose:
                elapsed = time.perf_counter() - _start
                if pre_text != None:
                    print(pre_text)
                st = f"Finished {text} in:"
                ed = f"{np.round(elapsed, 1)}"
                print(f"{st:<60}{ed:>10} {'seconds':<10}")
                if post_text != None:
                    print(post_text)
            return out
        return wrapped
    return wrapper

def complexity(data):
    data = data.astype(int)
    string = "".join([str(x) for x in data])
    c = bz2.compress(str.encode(string))
    return len(string)/len(c)


def gen_curve(ps,pe,t=np.linspace(0, 1, 100)):
    '''Created with the help of Asaf Soreq '''
    cp,dist = create_ctrl(ps,pe)
    if np.any(ps != pe):
        t = np.array([t,t])
        return (1-t)**2*ps.reshape(2,1) + 2*(1-t)*t*cp.reshape(2,1)+t**2*pe.reshape(2,1)
    else: 
        t = np.linspace(0,2*np.pi,100)
        return dist * np.array([np.sin(t),np.cos(t)])+cp.reshape(2,1)


def create_ctrl(p1,p2,w=0.5,f=1):
    '''Created with the help of Asaf Soreq'''
    d = (p2-p1)
    if np.all(d==0):d=d+1
    dist = np.sqrt(d@d.T)*w # calc the distance between points 
    mp = (p1 + p2) / 2 # calc midpoint
    pg = np.arctan(-1/ np.divide(*(d))) # calc perpendicular gradient
    return mp + f * (dist * np.array([np.sin(pg),np.cos(pg)])),dist         


def bspline(cv, n=100, degree=3):
    """ Calculate n samples on a bspline
        source : https://stackoverflow.com/questions/28279060/splines-with-python-using-control-knots-and-endpoints
        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

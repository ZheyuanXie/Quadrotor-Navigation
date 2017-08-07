from math import cos, sin, sqrt, pi
from numpy import vectorize
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy

# initial position of the drone
init_x = 0.0
init_y = 0.0
init_z = 0.0


# defulat path is a circle 1m above ground with a radius of r
class default_path():
    def __init__(self, loc = (0,0,1) , r = 1):
        self.loc = loc
        self.r = r

        # vectorized function of the path
        self.x = vectorize(self.p_x)
        self.y = vectorize(self.p_y)
        self.z = vectorize(self.p_z)
        self.vx = vectorize(self.p_vx)
        self.vy = vectorize(self.p_vy)
        self.vz = vectorize(self.p_vz)

    def p_x(self,t):
        return self.loc[0] + self.r * cos(pi / 5 * t)

    def p_y(self,t):
        return self.loc[1] + self.r * sin(pi / 5 * t)

    def p_z(self,t):
        return self.loc[2]

    def p_vx(self,t):
        return self.r * 0.2 * pi * sin(pi/5*t)

    def p_vy(self,t):
        return self.r * 0.2 * pi * cos(pi/5*t)

    def p_vz(self,t):
        return 0

# This class initialize a path that 
class real_time_path():
    def __init__(self, target, speed = 0.1, t_start = 0, given_height = True):
        self.target = target
        self.speed = speed
        self.t_start = t_start
        self.given_height = given_height

        self.distance = sqrt(target[0]**2 + target[1]**2 + target[2]**2)
        self.estimate_time = self.distance / self.speed

        # vectorized function of the path
        self.x = vectorize(self.p_x)
        self.y = vectorize(self.p_y)
        self.z = vectorize(self.p_z)
        self.vx = vectorize(self.p_vx)
        self.vy = vectorize(self.p_vy)
        self.vz = vectorize(self.p_vz)
    
    def p_x(self,t):
        if self.distance == 0:
            return 0
        return self.target[0] * ((t - self.t_start) * self.speed / self.distance) * 2

    def p_y(self,t):
        if self.distance == 0:
            return 0
        return self.target[1] * ((t - self.t_start) * self.speed / self.distance)

    def p_z(self,t):
        if self.given_height:
            return 1
        if self.distance == 0:
            return 0
        return self.target[2] * ((t - self.t_start) * self.speed / self.distance) * 4

    def p_vx(self,t):
        if self.distance == 0:
            return 0
        else:
            return self.speed * (self.target[0] / self.distance)

    def p_vy(self,t):
        if self.distance == 0:
            return 0
        else:
            return self.speed * (self.target[1] / self.distance)

    def p_vz(self,t):
        if self.given_height:
            return 0
        if self.distance == 0:
            return 0
        else:
            return self.speed * (self.target[2] / self.distance)

# this class initialize a quintic path from (0,0,0) to (x,y,z) with in time T
class quintic_path():
    def __init__(self, vel, acc, tgt, speed = 0.1, t_dep = 0, given_height = True):
        self.vel = vel  # (vx,vy,vz) @ t = 0
        self.acc = acc  # (ax,ay,az) @ t = 0
        self.tgt = tgt  # (x,y,z)
        self.avg_speed = speed
        self.t_dep = t_dep
        self.t_arr = sqrt(tgt[0]**2 + tgt[1]**2 + tgt[2]**2) / self.avg_speed
        self.given_height = given_height

        # Now we calculate the coefficient for the path
        # For i = x,y,z : (a_i)t^5+(b_i)t^4+(c_i)t^3+(d_i)t^2+(e_i)t+f=0
        t = self.t_arr
        a = np.array([[t**5, t**4, t**3],
            [5*t**4, 4*t**3, 3*t**2],
            [20*t**3, 12*t**2, 6*t]])
        b_x = np.array([-0.5*acc[0]*t**2 - vel[0]*t + tgt[0], -acc[0]*t + vel[0], -acc[0]])
        b_y = np.array([-0.5*acc[1]*t**2 - vel[1]*t + tgt[1], -acc[1]*t + vel[1], -acc[1]])
        b_z = np.array([-0.5*acc[2]*t**2 - vel[2]*t + tgt[2], -acc[2]*t + vel[2], -acc[2]])
        self.coeff_x = np.hstack([np.linalg.solve(a, b_x), np.array([0.5 * acc[0], vel[0], 0])])
        self.coeff_y = np.hstack([np.linalg.solve(a, b_y), np.array([0.5 * acc[1], vel[1], 0])])
        self.coeff_z = np.hstack([np.linalg.solve(a, b_z), np.array([0.5 * acc[2], vel[2], 0])])

        # vectorized function of the path
        self.x = vectorize(self.p_x)
        self.y = vectorize(self.p_y)
        self.z = vectorize(self.p_z)
        self.vx = vectorize(self.p_vx)
        self.vy = vectorize(self.p_vy)
        self.vz = vectorize(self.p_vz)

    def p_x(self,t):
        t = t - self.t_dep
        a = self.coeff_x[0]
        b = self.coeff_x[1]
        c = self.coeff_x[2]
        d = self.coeff_x[3]
        e = self.coeff_x[4]
        return a*t**5 + b*t**4 + c*t**3 + d*t**2 + e*t

    def p_y(self,t):
        t = t - self.t_dep
        a = self.coeff_y[0]
        b = self.coeff_y[1]
        c = self.coeff_y[2]
        d = self.coeff_y[3]
        e = self.coeff_y[4]
        return a*t**5 + b*t**4 + c*t**3 + d*t**2 + e*t

    def p_z(self,t):
        t = t - self.t_dep
        if self.given_height:
            return 1
        a = self.coeff_z[0]
        b = self.coeff_z[1]
        c = self.coeff_z[2]
        d = self.coeff_z[3]
        e = self.coeff_z[4]
        return a*t**5 + b*t**4 + c*t**3 + d*t**2 + e*t

    def p_vx(self,t):
        t = t - self.t_dep
        a = self.coeff_x[0]
        b = self.coeff_x[1]
        c = self.coeff_x[2]
        d = self.coeff_x[3]
        e = self.coeff_x[4]
        return 5*a*t**4 + 4*b*t**3 + 3*c*t**2 + 2*d*t + e

    def p_vy(self,t):
        t = t - self.t_dep
        a = self.coeff_y[0]
        b = self.coeff_y[1]
        c = self.coeff_y[2]
        d = self.coeff_y[3]
        e = self.coeff_y[4]
        return 5*a*t**4 + 4*b*t**3 + 3*c*t**2 + 2*d*t + e

    def p_vz(self,t):
        t = t - self.t_dep
        if self.given_height:
            return 0
        a = self.coeff_z[0]
        b = self.coeff_z[1]
        c = self.coeff_z[2]
        d = self.coeff_z[3]
        e = self.coeff_z[4]
        return 5*a*t**4 + 4*b*t**3 + 3*c*t**2 + 2*d*t + e



if __name__=="__main__":
    path = quintic_path(vel=(0,2,0),acc=(0,1,4),tgt=(1,1,1),speed=1,t_dep=0)
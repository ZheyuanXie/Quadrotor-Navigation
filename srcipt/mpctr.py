import numpy as np
from numpy import cos, sin
from path import default_path


class Mpc:
    dt = 0.1  # time interval for state space prediction
    dt1 = 0.01  # time interval of the control output calculation
    m = 1.52  # mass of the drone
    g = 9.81

    # state space model definition: dx = A * x + B * u
    A = np.matrix([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    A = A * dt + np.eye(6)
    B = np.matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    B = B / float(m) * dt
    Q = np.diag([150, 150, 15, 0, 0, 0])
    P = np.eye(3)

    # matrix for predictive control calculation
    A1 = np.vstack([A,A**2,A**3,A**4,A**5,A**6,A**7,A**8,A**9,A**10])
    B1 = np.vstack([
        np.hstack([B, np.zeros([6, 3 * 9])]),
        np.hstack([A * B, B, np.zeros([6, 3 * 8])]),
        np.hstack([A ** 2 * B, A * B, B, np.zeros([6, 3 * 7])]),
        np.hstack([A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 6])]),
        np.hstack([A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 5])]),
        np.hstack([A ** 5 * B, A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 4])]),
        np.hstack([A ** 6 * B, A ** 5 * B, A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 3])]),
        np.hstack([A ** 7 * B, A ** 6 * B, A ** 5 * B, A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 2])]),
        np.hstack([A ** 8 * B, A ** 7 * B, A ** 6 * B, A ** 5 * B, A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B, np.zeros([6, 3 * 1])]),
        np.hstack([A ** 9 * B, A ** 8 * B, A ** 7 * B, A ** 6 * B, A ** 5 * B, A ** 4 * B, A ** 3 * B, A ** 2 * B, A * B, B])
    ])
    Q1 = np.diag(
        [150, 15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150,
         15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150, 15, 15, 10, 10, 10, 150, 15, 15,
         10, 10, 10, 150, 15, 15, 10, 10, 10])
    P1 = np.eye(30)

    def __init__(self):
        self.U = np.matrix([])
        self.step = 0  # loop between 0~4, 5 Us for every calc_U call (T ~= 0.1s)
        self.cnt = 0  # loop between 0~9, 10 calculations for every U  (T ~= 0.01s)
        self.time = 0  # timestamp of the last U calculation

    # Calculate U of the ss model, supposed to execute every 0.5s (2Hz)
    def calc_U(self, x, y, z, vx, vy, vz, t, path = default_path(r = 0.2)):
        X0 = np.vstack([x, y, z, vx, vy, vz])
        ts = np.matrix([t + 0.1, t + 0.2, t + 0.3, t + 0.4, t + 0.5, t + 0.6, t + 0.7, t + 0.8, t + 0.9, t + 1])
        x1 = path.x(ts)
        y1 = path.y(ts)
        z1 = path.z(ts)
        vx1 = path.vx(ts)
        vy1 = path.vy(ts)
        vz1 = path.vz(ts)
        Xd = np.matrix([])
        for i in range(10):
            if i == 0:
                Xd = np.vstack([x1[0,i],y1[0,i],z1[0,i],vx1[0,i],vy1[0,i],vz1[0,i]])
            else:
                Xd = np.vstack([Xd,x1[0,i],y1[0,i],z1[0,i],vx1[0,i],vy1[0,i],vz1[0,i]])
        U = - np.linalg.inv(self.B1.transpose()*self.Q1*self.B1)*((self.Q1*self.B1).transpose())*(self.A1*X0-Xd)
        self.U = U
        self.step = 0
        self.cnt = 0
        self.time = t
        #print U[0],U[1],U[2]
        return U

    # Calculate Ctrl based on x4, x5, x6, supposed to execute every 0.01s (100Hz)
    def calc_ctrl(self, x4, x5, x6):
        U = self.U
        if U.shape[0] != 30:
            print 'Error: U should be calculated first'
            return
        i = 0

        F = np.matrix([[U[3 * i + 0, 0]],
                       [U[3 * i + 1, 0]],
                       [U[3 * i + 2, 0] + self.m * self.g]
                       ])
        #print F
        b3 = F / np.sqrt(F[0, 0] ** 2 + F[1, 0] ** 2 + F[2, 0] ** 2)
        temp = np.matrix([
            [0, -b3[2, 0], b3[1, 0]],
            [b3[2, 0], 0, -b3[0, 0]],
            [-b3[1, 0], b3[0, 0], 0]
        ]) * np.matrix([
            [np.cos(x6)],
            [np.sin(x6)],
            [0]
        ])
        b2 = temp / np.sqrt(temp[0, 0] ** 2 + temp[1, 0] ** 2 + temp[2, 0] ** 2)
        b1 = np.matrix([
            [0, -b2[2, 0], b2[1, 0]],
            [b2[2, 0], 0, -b2[0, 0]],
            [-b2[1, 0], -b2[0, 0], 0]
        ]) * b3
        # Rd = np.vstack([b1.transpose(),b2.transpose(),b3.transpose()]).transpose()
        R = np.matrix([
            [cos(x6) * cos(x5), -sin(x6) * cos(x4) + cos(x6) * sin(x5) * sin(x4),
             sin(x6) * sin(x4) + cos(x6) * cos(x4) * sin(x5)],
            [sin(x6) * cos(x5), cos(x6) * cos(x4) + sin(x6) * sin(x5) * sin(x4),
             -cos(x6) * sin(x4) + sin(x5) * sin(x6) * cos(x4)],
            [-sin(x5), cos(x5) * sin(x4), cos(x5) * cos(x4)]
        ])
        T = F.transpose() * R * np.matrix([[0], [0], [1]])
        T = T[0, 0]
        x6d = 0
        x5d = np.arctan(-b1[2, 0] / (b1[0, 0] * cos(x6d) + b1[1, 0] * sin(x6d)))
        x4d = np.arctan((b3[0, 0] * sin(x6d) - b3[1, 0] * cos(x6d)) / (b2[1, 0] * cos(x6d) - b2[0, 0] * sin(x6d)))

        #print 't=%f, roll=%f, pitch=%f, T=%f' % (self.time + self.step * 0.1 + self.cnt * 0.01, x4d, x5d, T)

        self.cnt += 1
        if self.cnt == 10:
            self.cnt = 0
            self.step += 1
        return (x4d/5,x5d/5,x6d/5,T)
        #return [np.rad2deg(x4d), np.rad2deg(x5d), np.rad2deg(x6d), T]

if __name__ == "__main__":
    m=Mpc()
    m.calc_U(0,0,0,0,0,0,0)
    for i in range(50):
        print m.calc_ctrl(0,0,0)

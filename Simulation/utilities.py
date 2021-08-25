import numpy as np
from sympy import Point3D, Plane, Line3D

def Reflection(V, N):
    R = V - 2 * N.T * (np.dot(V, N))
    return R


####################################################################################################################
# THIS FUNCTION IS SPECIFIC FOR THE SUN REFLECTION AND DETERMINING THE INTERSECTION BETWEEN THE VECTOR AND A PLANE #
####################################################################################################################
def Intersection(Plane, vector, pointFrom):
    d = np.sum(np.array(Plane) * np.array(vector))
    t0 = np.sum(np.array(Plane) * np.array(pointFrom))

    t = (Plane[3] - d)/t0

    vectorMoved = np.array(vector) * t

    position = pointFrom + vectorMoved

    return position

####################################################################################
# THIS FUNCTION IS USED TO DETERMINE WHETHER A POINT IS BETWEEN TWO PARALLEL LINES #
####################################################################################
def PointWithinParralelLines(Line1, Line2, Point):
    z1 = Line1[0]*Point[0] + Line1[1]
    z2 = Line2[0]*Point[0] + Line2[1]

    if (Point[1] < z1 and Point[1] > z2) or (Point[1] > z1 and Point[1] < z2):
        return True
    else:
        return False


def lineEquation(PointFrom, PointTo):
    m = (PointTo[0] - PointFrom[0])/(PointTo[1] - PointFrom[1])

if __name__ == '__main__':
    a = Plane(Point3D(1/2, 1/2, -1/2), Point3D(1/2, 0, 0), Point3D(1/2, -1/2, -1/2))

    print(a.equation())

    a = Plane(Point3D(1/2, 0, 0), normal_vector=(1/2, 0, 0))

    print(a.equation())

    a.intersection()

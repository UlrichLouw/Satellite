import numpy as np

def Reflection(V, N):
    R = V - 2 * N.T * (np.dot(V, N))
    R = R/np.linalg.norm(R)
    return R


####################################################################################################################
# THIS FUNCTION IS SPECIFIC FOR THE SUN REFLECTION AND DETERMINING THE INTERSECTION BETWEEN THE VECTOR AND A PLANE #
####################################################################################################################
def Intersection(Plane, vector, pointFrom):
    vector = np.array(vector)
    d = np.sum(np.array(Plane[:3]) * vector)
    t0 = np.sum(np.array(Plane[:3]) * np.array(pointFrom))

    t = (Plane[3] - d)/t0

    vectorMoved = vector * t

    position = pointFrom + vectorMoved

    return position

####################################################################################
# THIS FUNCTION IS USED TO DETERMINE WHETHER A POINT IS BETWEEN TWO PARALLEL LINES #
####################################################################################
def PointWithinParallelLines(Line1, Line2, Point):
    z1 = Line1[0]*Point[0] + Line1[1]
    z2 = Line2[0]*Point[0] + Line2[1]

    if (Point[1] < z1 and Point[1] > z2) or (Point[1] > z1 and Point[1] < z2):
        return True
    else:
        return False

################################
# FIND THE EQUATION FOR A LINE #
################################
def lineEquation(PointFrom, PointTo):
    m = (PointTo[0] - PointFrom[0])/(PointTo[1] - PointFrom[1])
    c = PointTo[0] - m*PointTo[1]
    return m, c

################################
# FIND THE EQUATION FOR A LINE #
################################
def line2Equation(PointFrom, PointTo):
    m = (PointTo[1] - PointFrom[1])/(PointTo[0] - PointFrom[0])
    c = PointTo[1] - m*PointTo[0]
    return m, c
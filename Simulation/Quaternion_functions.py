import numpy as np

def rad2deg(rad):
    return rad / np.pi * 180

def deg2rad(deg):
    return deg / 180 * np.pi

def A_to_quaternion(A):
    qw = 0.5 * np.sqrt(1 + A[0,0] + A[1,1] + A[2,2])
    qx = 1/(4*qw) * (A[1,2] - A[2,1])
    qy = 1/(4*qw) * (A[2,0] - A[0,2])
    qz = 1/(4*qw) * (A[0,1] - A[1,0])
    return np.array(([qx, qy, qz, qw]))

def euler_to_quaternion(roll, pitch, yaw):
    # This function is used to translate the euler angles (roll, pitch, yaw) to quaternions
    roll, pitch, yaw = [deg2rad(roll), deg2rad(pitch), deg2rad(yaw)]
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array(([qx, qy, qz, qw]))

def vectorsToQuaternion(vectorRef, vectorCurrent):
    Length1 = np.sqrt(np.sum(vectorCurrent**2))
    Length2 = np.sqrt(np.sum(vectorRef**2))
    vectorA = np.cross(vectorCurrent, vectorRef)
    qx, qy, qz = vectorA
    qw = np.sqrt((Length1**2)*(Length2**2)) + np.dot(vectorCurrent, vectorRef)
    q = np.array(([qx, qy, qz, qw]))
    q = q/np.linalg.norm(q)
    return q


def quaternion_error(current_quaternion, command_quaternion):
    ####################################################################################################
    # FOR THE CONTROL OF THE ADCS THE CONTROL SYSTEM WILL PROVIDE A COMMAND QUATERNION. THE DIFFERENCE #
    # BETWEEN THE CURRENT QUATERNION AND THE COMMAND QUATERNION IS REQUIRED TO PRODUCE A CHANGE IN THE #
    #                                             SYSTEM.                                              #
    ####################################################################################################

    qc1, qc2, qc3, qc4 = command_quaternion
    q_c = np.array(([[qc4, qc3, -qc2, -qc1],
                    [-qc3, qc4, qc1, -qc2],
                    [qc2, -qc1, qc4, -qc3], 
                    [qc1, qc2, qc3, qc4]]))
    
    q_error = q_c @ current_quaternion

    q_error = q_error / np.linalg.norm(q_error)

    return q_error
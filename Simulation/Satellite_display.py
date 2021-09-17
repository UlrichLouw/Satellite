import Simulation.Satellite_body as sat_body
import pygame
from operator import itemgetter
import sys
import matplotlib
from Simulation.Parameters import SET_PARAMS
   
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

zoom_out_factor = 2

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """
    def __init__(self, width, height, sat_body, number_of_satellites = 1):
        self.Radius_earth = SET_PARAMS.Radius_earth/1000
        self.fault = "None"
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlim3d(-self.Radius_earth*zoom_out_factor, self.Radius_earth*zoom_out_factor)
        self.ax.set_ylim3d(-self.Radius_earth*zoom_out_factor, self.Radius_earth*zoom_out_factor)
        self.ax.set_zlim3d(-self.Radius_earth*zoom_out_factor, self.Radius_earth*zoom_out_factor)
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.width = width
        self.height = height
        self.sat_body = sat_body
        self.position_ = []
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Satellite rotating around the earth')
        self.background = (10,10,50)
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.angle = 0
        self.step = 0
        self.sun_in_view = False
        self.current_positions = [None] * number_of_satellites
        self.number_of_satellites = number_of_satellites
        self.cmap = matplotlib.cm.get_cmap('nipy_spectral')

    def run(self,w, q, A, r,sun_in_view, only_positions = False, sat_num = 1):
        """ Create a pygame screen until it is closed. """
        running = True
        loopRate = 50
        angularVeloctiy = w
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        self.clock.tick(loopRate)
        self.display(q,A,r,sun_in_view, only_positions, sat_num)
        pygame.display.flip()

    def stop(self):
        pygame.quit()
        #sys.exit()

    def display(self,q,A,r,sun_in_view, only_positions, sat_num):
        """ Draw the wireframes on the screen. """
        r = r/1000
        self.screen.fill(self.background)

        if not only_positions:
            # Get the current attitude
            yaw, pitch, roll = self.sat_body.getAttitude(q)
            self.messageDisplay("Yaw: %.1f" % yaw,
                                self.screen.get_width()*0.75,
                                self.screen.get_height()*0,
                                (220, 20, 60))      # Crimson
            self.messageDisplay("Pitch: %.1f" % pitch,
                                self.screen.get_width()*0.75,
                                self.screen.get_height()*0.05,
                                (0, 255, 255))     # Cyan
            self.messageDisplay("Roll: %.1f" % roll,
                                self.screen.get_width()*0.75,
                                self.screen.get_height()*0.1,
                                (65, 105, 225))    # Royal Blue

            # Transform nodes to perspective view
            dist = 5
            pvNodes = []
            pvDepth = []
            for node in self.sat_body.nodes:
                point = [node.x, node.y, node.z]
                newCoord = self.sat_body.rotatePoint(point, q)
                comFrameCoord = self.sat_body.convertToComputerFrame(newCoord)
                pvNodes.append(self.projectOthorgraphic(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
                                                        self.screen.get_width(), self.screen.get_height(),
                                                        70, pvDepth))
                """
                pvNodes.append(self.projectOnePointPerspective(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
                                                            self.screen.get_width(), self.screen.get_height(),
                                                            5, 10, 30, pvDepth))
                """

            # Calculate the average Z values of each face.
            avg_z = []
            for face in self.sat_body.faces:
                n = pvDepth
                z = (n[face.nodeIndexes[0]] + n[face.nodeIndexes[1]] +
                    n[face.nodeIndexes[2]] + n[face.nodeIndexes[3]]) / 4.0
                avg_z.append(z)
            # Draw the faces using the Painter's algorithm:
            for idx, val in sorted(enumerate(avg_z), key=itemgetter(1)):
                face = self.sat_body.faces[idx]
                pointList = [pvNodes[face.nodeIndexes[0]],
                            pvNodes[face.nodeIndexes[1]],
                            pvNodes[face.nodeIndexes[2]],
                            pvNodes[face.nodeIndexes[3]]]
                pygame.draw.polygon(self.screen, face.color, pointList)
        self.plot(r,sun_in_view, sat_num, only_positions)

    # One vanishing point perspective view algorithm
    def projectOnePointPerspective(self, x, y, z, win_width, win_height, P, S, scaling_constant, pvDepth):
        # In Pygame, the y axis is downward pointing.
        # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
        # This will result in y' = -y and z' = -z
        xPrime = x
        yPrime = -y
        zPrime = -z
        xProjected = xPrime * (S/(zPrime+P)) * scaling_constant + win_width / 2
        yProjected = yPrime * (S/(zPrime+P)) * scaling_constant + win_height / 2
        pvDepth.append(1/(zPrime+P))
        return (round(xProjected), round(yProjected))

    # Normal Projection
    def projectOthorgraphic(self, x, y, z, win_width, win_height, scaling_constant, pvDepth):
        # In Pygame, the y axis is downward pointing.
        # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
        # This will result in y' = -y and z' = -z
        xPrime = x
        yPrime = -y
        xProjected = xPrime * scaling_constant + win_width / 2
        yProjected = yPrime * scaling_constant + win_height / 2
        # Note that there is no negative sign here because our rotation to computer frame
        # assumes that the computer frame is x-right, y-up, z-out
        # so this z-coordinate below is already in the outward direction
        pvDepth.append(z)
        return (round(xProjected), round(yProjected))

    def messageDisplay(self, text, x, y, color):
        textSurface = self.font.render(text, True, color, self.background)
        textRect = textSurface.get_rect()
        textRect.topleft = (x, y)
        self.screen.blit(textSurface, textRect)

    def plot(self,r,sun_in_view, sat_num, positions_only = False):
        if self.sun_in_view != sun_in_view or self.step == 0:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u)*np.sin(v)*(self.Radius_earth)
            y = np.sin(u)*np.sin(v)*(self.Radius_earth)
            z = np.cos(v)*(self.Radius_earth)
            if sun_in_view:
                self.ax.plot_wireframe(x, y, z, color="y", alpha = 0.2)
            else:
                self.ax.plot_wireframe(x, y, z, color="b", alpha = 0.2)
            self.sun_in_view = sun_in_view
        
        self.step += 1

        if positions_only:
            self.current_positions[sat_num] = r
            i = 0
            for sat_pos in self.current_positions:
                i += 1
                try:
                    position = np.array((sat_pos))
                    x = position[0]
                    y = position[1]
                    z = position[2]
                    self.ax.plot(x,y,z, color=self.cmap(i/self.number_of_satellites), marker=".", alpha = 0.1)
                except:
                    pass

        else:
            self.position_.append(r)
            position = np.array((self.position_))
            x = position[:,0]
            y = position[:,1]
            z = position[:,2]
            if self.step%100 == 0:
                self.position_ = self.position_[int(len(self.position_)/2):-1]
            #self.ax.plot(x, y, z, color="k")
            if self.fault == "None":
                self.ax.plot(x[-1],y[-1],z[-1], color="b", marker=".", alpha = 0.1)
            else:
                self.ax.plot(x[-1],y[-1],z[-1], color="r", marker=".", alpha = 0.05)

        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        self.ax.view_init(15, self.angle) #ax.view_init(30, angle)
        self.angle += 1
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()
        self.screen.blit(pygame.image.fromstring(raw_data, size, "RGB"), (0,0))

    def save_plot(self, fault):
        path_to_folder = Path("Orbit_3D")
        path_to_folder.mkdir(exist_ok=True)
        self.ax.view_init(15,2450)
        plt.savefig("Orbit_3D/" + str(fault) + ".png")
        

def initializeCube(Dimensions):
    width, length, height = Dimensions*5

    block = sat_body.Wireframe()

    block_nodes = [(x, y, z) for x in (-width, width) for y in (-height, height) for z in (-length, length)]
    node_colors = [(255, 255, 255)] * len(block_nodes)
    block.addNodes(block_nodes, node_colors)
    #block.outputNodes()

    faces = [(0, 2, 6, 4), (0, 1, 3, 2), (1, 3, 7, 5), (4, 5, 7, 6), (2, 3, 7, 6), (0, 1, 5, 4)]
    colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
    block.addFaces(faces, colors)
    #block.outputFaces()

    return block


if __name__ == '__main__':
    pass
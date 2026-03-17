from OpenGL.GL import GL_COLOR_BUFFER_BIT
from OpenGL.GL import GL_DEPTH_BUFFER_BIT
from OpenGL.GL import GL_DEPTH_TEST
from OpenGL.GL import glClear
from OpenGL.GL import glEnable
from OpenGL.GL import glLoadIdentity
from OpenGL.GL import glRotatef
from OpenGL.GL import glTranslatef
from OpenGL.GLUT import GLUT_DEPTH
from OpenGL.GLUT import GLUT_DOUBLE
from OpenGL.GLUT import GLUT_RGB
from OpenGL.GLUT import glutCreateWindow
from OpenGL.GLUT import glutDisplayFunc
from OpenGL.GLUT import glutIdleFunc
from OpenGL.GLUT import glutInit
from OpenGL.GLUT import glutInitDisplayMode
from OpenGL.GLUT import glutInitWindowSize
from OpenGL.GLUT import glutMainLoop
from OpenGL.GLUT import glutSwapBuffers
from OpenGL.GLUT import glutWireCube

GL_COLOR_BUFFER_BIT_VALUE = int(GL_COLOR_BUFFER_BIT)
GL_DEPTH_BUFFER_BIT_VALUE = int(GL_DEPTH_BUFFER_BIT)
GL_DEPTH_TEST_VALUE = int(GL_DEPTH_TEST)
GLUT_DEPTH_VALUE = int(GLUT_DEPTH)
GLUT_DOUBLE_VALUE = int(GLUT_DOUBLE)
GLUT_RGB_VALUE = int(GLUT_RGB)

rotation = 0


def draw_cube():
    glutWireCube(2)


def render():
    global rotation

    glClear(GL_COLOR_BUFFER_BIT_VALUE | GL_DEPTH_BUFFER_BIT_VALUE)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -5)
    glRotatef(rotation, 1, 1, 1)

    draw_cube()

    glutSwapBuffers()
    rotation += 1


def start():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE_VALUE | GLUT_RGB_VALUE | GLUT_DEPTH_VALUE)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"3D Gesture Visualizer")

    glEnable(GL_DEPTH_TEST_VALUE)

    glutDisplayFunc(render)
    glutIdleFunc(render)
    glutMainLoop()

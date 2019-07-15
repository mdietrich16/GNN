# -*- coding: utf-8 -*-

import numpy as np
from glumpy import gloo, gl, app, transforms, data
from glumpy.geometry import primitives
from glumpy.graphics.collections import GlyphCollection
from glumpy.graphics.text import FontManager

from trainer import SGD, NAdam, Adam, Momentum, RMSProp, AdaMax, AdaGrad

window = app.Window(800, 600, color=(.2,.2,.35,1))

vertex = """
attribute vec3 position;
void main()
{
     gl_Position = <transform(position)>;
     <viewport.transform>;
}
"""

fragment = """
uniform vec4 color;
void main()
{
     gl_FragColor = color;
}
"""

surface_vertex = """
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
varying vec2 v_texcoord;
varying vec3 v_normal;
varying vec3 v_position;
void main()
{
     v_texcoord = texcoord;
     v_normal = normal;
     v_position = position.xyz;
     gl_Position = <transform(position)>;
     <viewport.transform>;
}
"""

surface_fragment = """
uniform vec4 color;
uniform sampler2D texture;
uniform vec3 light_position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 normal;
varying vec2 v_texcoord;
varying vec3 v_normal;
varying vec3 v_position;
float lighting(vec3 v_normal, vec3 light_position)
{
     vec3 n = normalize(normal * vec4(v_normal,1.0)).xyz;
     vec3 position = vec3(view * model * vec4(v_position, 1));
     vec3 surface_to_light = light_position - position;
     float brightness = dot(n, surface_to_light) /
         (length(surface_to_light) * length(n));
     brightness = max(min(brightness,1.0),0.0);
     return brightness;
}
void main()
{
     float c = 0.6 + 0.4*texture2D(texture, v_texcoord).r;
     gl_FragColor = color * vec4(c, c, c, 1);
     //* (0.5 + 0.5*lighting(v_normal, light_position));
}"""

def interp(pos, factor=4, axis=1):
    if not isinstance(pos, np.ndarray):
        raise TypeError('pos has to be ndarray')
    if len(pos.shape) < axis:
        raise IndexError('axis bigger than array dimensionality')
    factor = int(round(factor))
    l = pos.shape[axis]
    ix = [slice(None)] * len(pos.shape)
    ix[axis] = slice(0, 1)
    o = np.repeat(pos[ix], (l-1)*factor+1, axis=axis)
    ix = [slice(None)] * len(pos.shape)
    ix[axis] = slice(1, None)
    o[ix] += np.cumsum(np.repeat(np.diff(pos, axis=axis)/factor, factor, axis=axis), axis=axis)
    return o


def func1(pos):
    return np.square(pos).dot(np.array([[1.], [-1]]))


def func2(pos):
    return -np.exp(np.sum(-1*np.square(pos), axis=-1, keepdims=True))


def func3(pos):
    return (1 - pos.dot([[1.], [0.]]) / 2 + np.sum(np.power(pos, [5., 3.]))) * np.exp(np.sum(-1*np.square(pos)))


def grad1(pos):
    return pos * [[2, -2]]


def grad2(pos):
    return 2. * np.exp(np.sum(-1*np.square(pos), axis=-1, keepdims=True)) * pos


def grad3(pos):
    return np.ones_like(pos)


func = func1
grad = grad1


tries = 2
steps = 200
kernels = np.array([SGD, NAdam, RMSProp, AdaMax, AdaGrad])
k = len(kernels)
caches = [(), ] * k
hp = [ker.__defaults__[0][1:] for ker in kernels]
lrs = [ker.__defaults__[0][0] for ker in kernels]
factors = np.logspace(np.log10(np.min(lrs))-1, np.log10(np.max(lrs))+1, tries)
pos_ = np.array([[1., 0.]]) + np.random.randn(1, 2)*0.01
print('Starting pos for all optimizers: [{:6.4}, {:6.4}]'.format(pos_[0, 0], pos_[0, 1]))

b_pos = np.zeros((k, steps, 1, 2))
b_fac = np.ones(k)
best = np.full(k, np.inf)

for l in range(tries):
    positions = np.zeros((k, steps, 1, 2))
    positions[:, 0] = pos_
    caches = [(), ] * k
    for t in range(steps-1):
        dp = grad(positions[:, t])
        for i in range(k):
            pos = positions[i, t].copy()
            caches[i] = kernels[i](pos[:], dp[i], hp=(factors[l], *hp[i]), cache=caches[i])
            positions[i, t+1] = pos[:]
    now = np.min(func(positions), axis=1).ravel()
    better = now < best
    if np.any(better):
        b_pos[better] = positions[better]
        best[better] = now[better]
        b_fac[better] = factors[l]
print('Sorted Optimizers: {}'.format([k.__name__ for k in kernels[np.argsort(best)]]))
np.clip(b_pos, -4., 4., out=positions)
n = 64
vertices, s_indices = primitives.plane(2*np.maximum(-np.min(positions), np.max(positions)) + 0.2, n=n)
zoom = (2 * (2*np.maximum(-np.min(positions), np.max(positions)) + 0.2)**2)
Ix = []
for i in range(n): Ix.append(i)
for i in range(1,n): Ix.append(n-1+i*n)
for i in range(n-1): Ix.append(n*n-1-i)
for i in range(n-1): Ix.append(n*(n-1) - i*n)
b_indices = np.array(Ix, dtype=np.uint32).view(gloo.IndexBuffer)

Z = func(vertices['position'][:4096, :2])
z_min, z_ptp = Z.min(), Z.ptp()
Z = (Z-z_min)/z_ptp
vertices['position'][:4096, 2] = Z.ravel()

normals = np.empty(4096, dtype=[('normal', np.float32, 3)])
normals['normal'] = np.array([[-g[0], -g[1], 1.] for g in grad(vertices['position'][:4096, :2])], dtype=np.float32)

steps = (steps-1)*4+1
positions = interp(positions)
pos_ = np.concatenate((positions, (func(positions)-z_min)/z_ptp),axis=3).reshape(k, -1, 3)
positions = np.empty((k, steps), dtype=[('position', np.float32, 3)])
positions['position'] = pos_
del pos_

colors_ = np.array([[(i/k) % 1, (0.5+i/k) % 1, (1-i/k) % 1] for i in range(k)])
colors = np.empty((k), dtype=[('color', np.float32, 3)])
colors['color'] = colors_
del colors_

viewport = transforms.Viewport()
transform = (transforms.Trackball(transforms.Position()))

regular = FontManager.get("OpenSans-Regular.ttf", size=12, mode="agg")
labels = GlyphCollection("agg", transform=transform, viewport=viewport, color="shared")
for i, j in enumerate(kernels):
    labels.append(j.__name__, regular, origin=(0., 0., 2.+i/6), color=(*colors['color'][i], 1.))

balls = gloo.Program("""
                       attribute vec3 position;
                       attribute vec3 color;

                       uniform vec3 light_position;
                       uniform float radius;

                       varying vec4 v_eye_position;
                       varying vec3 v_light_direction;
                       varying vec3 v_color;
                       varying float v_radius;
                       varying float v_size;
                       void main()
                       {
                            v_color = color;
                            v_radius = radius;
                            v_eye_position = <transform.trackball_view> *
                            <transform.trackball_model> *
                            vec4(position,1.0);
                            v_light_direction = normalize(light_position);
                            gl_Position = <transform(position)>;
                            vec4 p = <transform.trackball_projection> *
                            vec4(radius, radius, v_eye_position.z, v_eye_position.w);
                            v_size = 512.0 * p.x / p.w;
                            gl_PointSize = v_size + 5.0;
                            <viewport.transform>;
                       }
                       """, """
                       #include "antialias/outline.glsl"



                       varying vec4 v_eye_position;
                       varying vec3 v_light_direction;
                       varying vec3 v_color;
                       varying float v_size;
                       varying float v_radius;

                       void main()
                       {
                            vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
                            float point_size = v_size + 5.0;
                            float distance = length(P * point_size) - v_size/2;

                            vec2 texcoord = gl_PointCoord * 2. - vec2(1.0);
                            float d = 1.0 - texcoord.x*texcoord.x - texcoord.y*texcoord.y;

                            if (d <= 0.0) discard;

                            float z = sqrt(d);
                            vec4 pos = v_eye_position;
                            pos.z += v_radius * z;
                            vec3 pos2 = pos.xyz;

                            pos = <transform.trackball_projection> * pos;
                            gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
                            vec3 normal = vec3(texcoord.x, texcoord.y, z);
                            float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);
                            vec4 color = vec4((0.5 + 0.5 * diffuse) * v_color, 1.0);
                            //gl_FragColor = outline(distance, 1.0, 1.0, vec4(0, 0, 0, 1), color);
                            gl_FragColor = color;
                       }""")


paths = [gloo.Program(vertex, fragment) for i in range(k)]
surface = gloo.Program(vertex=surface_vertex, fragment=surface_fragment)


balls.bind(positions[:, 0].ravel().view(gloo.VertexBuffer))
balls.bind(colors.ravel().view(gloo.VertexBuffer))
balls['radius'] = 0.05
balls['light_position'] = 0., 0., 2.
balls['transform'] = transform
balls['viewport'] = viewport

surface.bind(vertices)
surface["color"] = 1, 1, 1, 1
surface['texture'] = data.checkerboard(32, 24)

surface['transform'] = transform
surface['viewport'] = viewport

for i in range(k):
    paths[i].bind(positions[i].ravel()[:1].view(gloo.VertexBuffer))
    paths[i]['transform'] = transform
    paths[i]['viewport'] = viewport
    paths[i]['color'] = (0+i/k) % 1, (0.5+i/k) % 1, (1-i/k) % 1, 1


@window.event
def on_draw(dt):
    global n, steps, play, incr
    if play:
        n = (n+incr) % steps
    window.clear()

    balls.bind(positions[:, int(n)].ravel().view(gloo.VertexBuffer))

    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
    surface["color"] = 1, 1, 1, 1
    surface.draw(gl.GL_TRIANGLES, s_indices)

    gl.glEnable(gl.GL_BLEND)
    labels.draw()
    balls.draw(gl.GL_POINTS)
    gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
    gl.glDepthMask(gl.GL_FALSE)
    surface["color"] = 0, 0, 0, 1
    surface.draw(gl.GL_LINE_LOOP, b_indices)

    for i in range(k):
        paths[i].bind(positions[i].ravel()[:int(n)].view(gloo.VertexBuffer))
        paths[i].draw(gl.GL_LINE_STRIP)

    gl.glDepthMask(gl.GL_TRUE)
#    model = surface['transform']['model'].reshape(4,4)
#    view  = surface['transform']['view'].reshape(4,4)
#    surface['view']  = view
#    surface['model'] = model
#    surface['normal'] = np.array(np.matrix(np.dot(view, model)).I.T)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glPolygonOffset(5, 5)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glLineWidth(5)
    reset()


@window.event
def on_key_press(key, modifiers):
    global play
    if key == app.window.key.SPACE:
        reset()


@window.event
def on_character(character):
    global play, n, incr
    if character.capitalize() == 'P':
        play = not play
        print(n)
    elif character.capitalize() == 'R':
        n = 0.
        incr = 0.5
    elif character == '+':
        incr *= 2.
    elif character == '-':
        incr *= 0.5


def reset():
    global zoom
    transform.theta = 45
    transform.phi = -10
    transform.zoom = zoom


n = 0.
incr = 0.5
play = False
window.attach(transform)
window.attach(viewport)
#window.attach(labels['viewport'])
app.run(framerate=60)

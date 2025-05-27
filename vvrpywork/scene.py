'''Implements objects to represent 2D and 3D scenes containing shapes.'''

from beartype.vale import Is
from beartype.typing import Callable
import ctypes
from inspect import getfullargspec
from io import StringIO
from numbers import Number
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import pyglet
from typing import Annotated, Any

from vvrpywork.constants import Key
# from vvrpywork.shapes.abstract import Shape, ShapeSet

if os.name == "nt":
    ctypes.windll.user32.SetProcessDPIAware()


NDArray3 = Annotated[np.ndarray, Is[lambda array: array.shape == (3,)]]
List3 = Annotated[list[Number], 3]
Tuple3 = tuple[Number, Number, Number]

Shape = Any
ShapeSet = Any

MouseType = Any
KeyType = Any
ModifierType = Any


class Scene2D:
    '''A class representing a 2D Scene.
    
    This class is meant to be inherited in order to create your own 2D scene.
    Calling Scene2D.__init__ will set up the scene and running
    Scene2D.mainloop will display a window with the scene. There are several
    methods that act as event listeners and can be overridden by the subclass.
    '''
    def __init__(self, width:int, height:int, caption:None|str=None, resizable:bool=False):
        '''Initializes a 2D Scene.

        Args:
            width: The width of the window, in pixels.
            height: The height of the window, in pixels.
            caption: The title of the window.
            resizable: Whether the window is resizable.
        '''

        caption = "2D Scene" if caption is None else caption

        self._window = pyglet.window.Window(width, height, caption, resizable)
        pyglet.gl.glClearColor(0.7,0.7,0.7,1)
        self._window.view = pyglet.math.Mat4((width/200, 0, 0, 0, 0, height/200, 0, 0, 0, 0, 1, 0, width/2, height/2, 0, 1))

        self._shapeDict = {}
        self._shapeBatch = pyglet.graphics.Batch()
        self._layer = 0

        layout_vertex_source = """#version 330 core
            in vec3 position;
            in vec4 colors;
            in vec3 tex_coords;
            in vec3 translation;
            in vec3 view_translation;
            in vec2 anchor;
            in float rotation;
            in float visible;

            out vec4 text_colors;
            out vec2 texture_coords;
            out vec4 vert_position;

            uniform WindowBlock
            {
                mat4 projection;
                mat4 view;
            } window;

            mat4 m_rotation = mat4(1.0);
            vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
            mat4 m_anchor = mat4(1.0);
            mat4 m_translate = mat4(1.0);
            mat4 new_view = mat4(1.0);

            void main()
            {
                m_translate[3][0] = translation.x * 4.0f;
                m_translate[3][1] = translation.y * 4.0f;
                m_translate[3][2] = translation.z;

                m_rotation[0][0] =  cos(-radians(rotation));
                m_rotation[0][1] =  sin(-radians(rotation));
                m_rotation[1][0] = -sin(-radians(rotation));
                m_rotation[1][1] =  cos(-radians(rotation));

                new_view = window.view;
                new_view[0][0] = 1;
                new_view[1][1] = 1;

                gl_Position = window.projection * new_view * m_translate * m_anchor * m_rotation * vec4(position + view_translation + v_anchor, 1.0) * visible;

                vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
                text_colors = colors;
                texture_coords = tex_coords.xy;
            }
        """
        layout_fragment_source = """#version 330 core
            in vec4 text_colors;
            in vec2 texture_coords;
            in vec4 vert_position;

            out vec4 final_colors;

            uniform sampler2D text;
            uniform bool scissor;
            uniform vec4 scissor_area;

            void main()
            {
                final_colors = vec4(text_colors.rgb, texture(text, texture_coords).a * text_colors.a);
                if (scissor == true) {
                    if (vert_position.x < scissor_area[0]) discard;                     // left
                    if (vert_position.y < scissor_area[1]) discard;                     // bottom
                    if (vert_position.x > scissor_area[0] + scissor_area[2]) discard;   // right
                    if (vert_position.y > scissor_area[1] + scissor_area[3]) discard;   // top
                }
            }
        """

        self._text_shader = pyglet.gl.current_context.create_program((layout_vertex_source, "vertex"), (layout_fragment_source, "fragment"))

        self._window.on_draw = self.__on_draw
        self._window.on_mouse_press = self.__on_mouse_press
        self._window.on_mouse_drag = self.__on_mouse_drag
        self._window.on_mouse_release = self.__on_mouse_release
        self._window.on_key_press = self.__on_key_press
        self._window.on_key_release = self.__on_key_release

        self.__on_draw = self._window.event(self.__on_draw)
        self.__on_mouse_press = self._window.event(self.__on_mouse_press)
        self.__on_mouse_drag = self._window.event(self.__on_mouse_drag)
        self.__on_mouse_release = self._window.event(self.__on_mouse_release)
        self.__on_key_press = self._window.event(self.__on_key_press)
        self.__on_key_release = self._window.event(self.__on_key_release)

        pyglet.clock.schedule(lambda _: self.on_idle())

    def __on_draw(self):
        self._window.clear()
        self._shapeBatch.draw()

    def __on_mouse_press(self, x, y, button, modifiers):
        x = x * 2 / self._window.width - 1
        y = y * 2 / self._window.height - 1
        self.on_mouse_press(x, y, button, modifiers)

    def on_mouse_press(self, x:Number, y:Number, button:MouseType, modifiers:ModifierType):
        '''Is called when a mouse button is pressed.

        Args:
            x: X coordinate of the mouse.
            y: Y coordinate of the mouse.
            button: Mouse button pressed.
            modifiers: Modifier keys pressed.
        '''
        pass

    def __on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        x = x * 2 / self._window.width - 1
        y = y * 2 / self._window.height - 1
        self.on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_drag(self, x:Number, y:Number, dx:Number, dy:Number, buttons:MouseType, modifiers:ModifierType):
        '''Is called when the mouse moves while mouse buttons are being pressed.

        Args:
            x: X coordinate of the mouse at the start of the event.
            y: Y coordinate of the mouse at the start of the event.
            dx: Change in X coordinate.
            dy: Change in Y coordinate.
            buttons: Mouse buttons pressed.
            modifiers: Modifier keys pressed.
        '''
        pass

    def __on_mouse_release(self, x, y, button, modifiers):
        x = x * 2 / self._window.width - 1
        y = y * 2 / self._window.height - 1
        self.on_mouse_release(x, y, button, modifiers)

    def on_mouse_release(self, x:Number, y:Number, button:MouseType, modifiers:ModifierType):
        '''Is called when a mouse button is released.

        Args:
            x: X coordinate of the mouse.
            y: Y coordinate of the mouse.
            button: Mouse button released.
            modifiers: Modifier keys pressed.
        '''
        pass

    def __on_key_press(self, symbol, modifiers):
        if symbol == Key.ESCAPE:
            self._window.close()
        self.on_key_press(symbol, modifiers)

    def on_key_press(self, symbol:KeyType, modifiers:ModifierType):
        '''Is called when a key is pressed.

        Args:
            symbol: Key symbol pressed.
            modifiers: Modifier keys pressed.
        '''
        pass

    def __on_key_release(self, symbol, modifiers):
        self.on_key_release(symbol, modifiers)

    def on_key_release(self, symbol:KeyType, modifiers:ModifierType):
        '''Is called when a key is released.

        Args:
            symbol: Key symbol released.
            modifiers: Modifier keys pressed.
        '''
        pass

    def on_idle(self):
        '''Is scheduled to be called every frame.'''
        pass

    def mainLoop(self, max_fps:int=60):
        '''Starts the main loop of the 2D scene.

        Args:
            max_fps: Maximum frames per second.
        '''
        pyglet.app.run(interval=1/max_fps)

    def addShape(self, shape:Shape|ShapeSet, name:None|str=None):
        '''Adds a shape to the scene.

        Args:
            shape: The shape to add.
            name: Name of the shape.
        '''
        # check type of shape
        shape._addToScene(self, name)
        self._layer += 1

    def updateShape(self, name:str):
        '''Update a shape in the scene.

        Args:
            name: Name of the shape to update.
        '''
        if name in self._shapeDict:
            self._shapeDict[name]["class"]._update(self._shapeDict[name]["shape"], self)

    def removeShape(self, name:str):
        '''Removes a shape from the scene.

        Args:
            name: Name of the shape to remove.
        '''
        if name in self._shapeDict:
            # self._shapeDict[name].delete()
            del self._shapeDict[name]


class Scene3D:
    '''A class representing a 3D Scene.
    
    This class is meant to be inherited in order to create your own 3D scene.
    Calling Scene3D.__init__ will set up the scene and running
    Scene3D.mainloop will display a window with the scene. There are several
    methods that act as event listeners and can be overridden by the subclass.
    '''
    def __init__(self, width:int, height:int, caption:None|str=None, output:bool=False, n_sliders:int=0):
        '''Initializes a 3D Scene.

        Args:
            width: The width of the window, in pixels.
            height: The height of the window, in pixels.
            caption: The title of the window.
            output: Whether to display output text on the left side.
            n_sliders: The number of sliders to display on the left side.
        '''

        caption = "3D Scene" if caption is None else caption

        gui.Application.instance.initialize()

        self._window = gui.Application.instance.create_window(caption, width, height)
        w = self._window

        em = w.theme.font_size
        
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(w.renderer)
        self._scene_widget.scene.set_background((0.7, 0.7, 0.7, 1))

        self._scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        #set up camera
        bounds = self._scene_widget.scene.bounding_box
        center = bounds.get_center()
        self._scene_widget.look_at(center, center + [0, 0, 1.5], [0, 1, 0])

        self._scene_widget.scene.camera.set_projection(90, 1, 0.01, 3.75, rendering.Camera.FovType.Vertical)  # defaults except for near_plane


        if output or n_sliders > 0:
            self._scene_widget.frame = gui.Rect(200, w.content_rect.y,
                                            w.content_rect.width - 200, w.content_rect.height)
        else:
            self._scene_widget.frame = gui.Rect(w.content_rect.x, w.content_rect.y,
                                            w.content_rect.width, w.content_rect.height)

        self._scene_widget.scene.show_axes(True)

        if output or n_sliders > 0:
            gui_layout = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
            gui_layout.frame = gui.Rect(w.content_rect.x, w.content_rect.height - (n_sliders * 26 + (n_sliders + 1) * 0.5 * em),
                                        200, n_sliders * 26 + (n_sliders + 1) * 0.5 * em)
            gui_layout.background_color = gui.Color(0.1, 0.1, 0.1)


            self._sliders = []
            for i in range(n_sliders):
                slider = gui.Slider(gui.Slider.Type.DOUBLE)
                slider.set_limits(0, 1)
                slider.set_on_value_changed(lambda v, i=i: self.on_slider_change(i, v))
                gui_layout.add_child(slider)
                self._sliders.append(slider)


            text_layout = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
            text_layout.frame = gui.Rect(w.content_rect.x, w.content_rect.y,
                                        200, w.content_rect.height - (n_sliders * 26 + (n_sliders + 1) * 0.5 * em))
            text_layout.background_color = gui.Color(0.2, 0.2, 0.2)


            self._sio = StringIO()

            self._text_output = gui.Label("")
            self._text_output.text_color = gui.Color(1, 1, 1)
            text_layout.add_child(self._text_output)

            w.add_child(text_layout)
            w.add_child(gui_layout)
        w.add_child(self._scene_widget)

        self._scene_widget.set_on_mouse(lambda mouseEvent: self._mouseEventToFunction(mouseEvent))
        self._window.set_on_key(lambda keyEvent: gui.Application.instance.post_to_main_thread(self._window, lambda: self._keyEventToFunction(keyEvent)))
        self._window.set_on_tick_event(lambda: gui.Application.instance.post_to_main_thread(self._window, self.on_idle))
        # self._window.set_on_tick_event(self.on_idle)
        self._modifiers = gui.KeyModifier.NONE.value
        self._last_coords = np.array((0., 0.))

        self._shapeDict = {}

    def on_mouse_press(self, x:Number, y:Number, z:Number|Any, button:MouseType, modifiers:ModifierType):
        '''Is called when a mouse button is pressed.

        Is called when a mouse button is pressed. Use the world_space
        decorator if you wish to get the world space coordinates of the
        mouse.

        Args:
            x: X coordinate of the mouse, in screen space.
            y: Y coordinate of the mouse, in screen space.
            z: Is always numpy.inf.
            button: Mouse button pressed.
            modifiers: Modifier keys pressed.
        '''
        pass

    def on_mouse_drag(self, x:Number, y:Number, z:Number|Any, dx:Number, dy:Number, dz:Number|Any, buttons:MouseType, modifiers:ModifierType):
        '''Is called when the mouse moves while mouse buttons are being pressed.

        Args:
            x: X coordinate of the mouse at the start of the event, in screen space.
            y: Y coordinate of the mouse at the start of the event, in screen space.
            z: Is always numpy.inf.
            dx: Change in X coordinate, in screen space.
            dy: Change in Y coordinate, in screen space.
            dz: Is always np.inf.
            buttons: Mouse buttons pressed.
            modifiers: Modifier keys pressed.
        '''
        pass

    def on_mouse_release(self, x:Number, y:Number, z:Number|Any, button:MouseType, modifiers:ModifierType):
        '''Is called when a mouse button is released.

        Args:
            x: X coordinate of the mouse, in screen space.
            y: Y coordinate of the mouse, in screen space.
            z: Is always numpy.inf.
            button: Mouse button released.
            modifiers: Modifier keys pressed.
        '''
        pass

    def on_key_press(self, symbol:KeyType, modifiers:ModifierType):
        '''Is called when a key is pressed.

        Args:
            symbol: Key symbol pressed.
            modifiers: Modifier keys pressed.
        '''
        pass
    
    def on_key_release(self, symbol:KeyType, modifiers:ModifierType):
        '''Is called when a key is released.

        Args:
            symbol: Key symbol released.
            modifiers: Modifier keys pressed.
        '''
        pass
    
    def on_idle(self):
        '''Is scheduled to be called every frame.'''
        return False
    
    def on_slider_change(self, slider_id:int, value:float):
        '''Is called when a slider is changed.
        
        Args:
            slider_id: The id of the slider that was changed.
            value: The new value of the slider.
        '''
        return True
    
    def mainLoop(self):
        '''Starts the main loop of the 3D scene.'''
        gui.Application.instance.run()

    _key_to_symbol = {
        gui.KeyName.BACKSPACE: pyglet.window.key.BACKSPACE,
        gui.KeyName.TAB: pyglet.window.key.TAB,
        gui.KeyName.ENTER: pyglet.window.key.ENTER,
        gui.KeyName.ESCAPE: pyglet.window.key.ESCAPE,
        gui.KeyName.SPACE: pyglet.window.key.SPACE,
        gui.KeyName.EXCLAMATION_MARK: pyglet.window.key.EXCLAMATION,
        gui.KeyName.DOUBLE_QUOTE: pyglet.window.key.DOUBLEQUOTE,
        gui.KeyName.HASH: pyglet.window.key.HASH,
        gui.KeyName.DOLLAR_SIGN: pyglet.window.key.DOLLAR,
        gui.KeyName.PERCENT: pyglet.window.key.PERCENT,
        gui.KeyName.AMPERSAND: pyglet.window.key.AMPERSAND,
        gui.KeyName.QUOTE: pyglet.window.key.APOSTROPHE,
        gui.KeyName.LEFT_PAREN: pyglet.window.key.PARENLEFT,
        gui.KeyName.RIGHT_PAREN: pyglet.window.key.PARENRIGHT,
        gui.KeyName.ASTERISK: pyglet.window.key.ASTERISK,
        gui.KeyName.PLUS: pyglet.window.key.PLUS,
        gui.KeyName.COMMA: pyglet.window.key.COMMA,
        gui.KeyName.MINUS: pyglet.window.key.MINUS,
        gui.KeyName.PERIOD: pyglet.window.key.PERIOD,
        gui.KeyName.SLASH: pyglet.window.key.SLASH,
        gui.KeyName.ZERO: pyglet.window.key._0,
        gui.KeyName.ONE: pyglet.window.key._1,
        gui.KeyName.TWO: pyglet.window.key._2,
        gui.KeyName.THREE: pyglet.window.key._3,
        gui.KeyName.FOUR: pyglet.window.key._4,
        gui.KeyName.FIVE: pyglet.window.key._5,
        gui.KeyName.SIX: pyglet.window.key._6,
        gui.KeyName.SEVEN: pyglet.window.key._7,
        gui.KeyName.EIGHT: pyglet.window.key._8,
        gui.KeyName.NINE: pyglet.window.key._9,
        gui.KeyName.COLON: pyglet.window.key.COLON,
        gui.KeyName.SEMICOLON: pyglet.window.key.SEMICOLON,
        gui.KeyName.LESS_THAN: pyglet.window.key.LESS,
        gui.KeyName.EQUALS: pyglet.window.key.EQUAL,
        gui.KeyName.GREATER_THAN: pyglet.window.key.GREATER,
        gui.KeyName.QUESTION_MARK: pyglet.window.key.QUESTION,
        gui.KeyName.AT: pyglet.window.key.AT,
        gui.KeyName.LEFT_BRACKET: pyglet.window.key.BRACKETLEFT,
        gui.KeyName.BACKSLASH: pyglet.window.key.BACKSLASH,
        gui.KeyName.RIGHT_BRACKET: pyglet.window.key.BRACKETRIGHT,
        gui.KeyName.CARET: pyglet.window.key.ASCIICIRCUM,
        gui.KeyName.UNDERSCORE: pyglet.window.key.UNDERSCORE,
        gui.KeyName.BACKTICK: pyglet.window.key.GRAVE,
        gui.KeyName.A: pyglet.window.key.A,
        gui.KeyName.B: pyglet.window.key.B,
        gui.KeyName.C: pyglet.window.key.C,
        gui.KeyName.D: pyglet.window.key.D,
        gui.KeyName.E: pyglet.window.key.E,
        gui.KeyName.F: pyglet.window.key.F,
        gui.KeyName.G: pyglet.window.key.G,
        gui.KeyName.H: pyglet.window.key.H,
        gui.KeyName.I: pyglet.window.key.I,
        gui.KeyName.J: pyglet.window.key.J,
        gui.KeyName.K: pyglet.window.key.K,
        gui.KeyName.L: pyglet.window.key.L,
        gui.KeyName.M: pyglet.window.key.M,
        gui.KeyName.N: pyglet.window.key.N,
        gui.KeyName.O: pyglet.window.key.O,
        gui.KeyName.P: pyglet.window.key.P,
        gui.KeyName.Q: pyglet.window.key.Q,
        gui.KeyName.R: pyglet.window.key.R,
        gui.KeyName.S: pyglet.window.key.S,
        gui.KeyName.T: pyglet.window.key.T,
        gui.KeyName.U: pyglet.window.key.U,
        gui.KeyName.V: pyglet.window.key.V,
        gui.KeyName.W: pyglet.window.key.W,
        gui.KeyName.X: pyglet.window.key.X,
        gui.KeyName.Y: pyglet.window.key.Y,
        gui.KeyName.Z: pyglet.window.key.Z,
        gui.KeyName.LEFT_BRACE: pyglet.window.key.BRACELEFT,
        gui.KeyName.PIPE: pyglet.window.key.BAR,
        gui.KeyName.RIGHT_BRACE: pyglet.window.key.BRACERIGHT,
        gui.KeyName.TILDE: pyglet.window.key.ASCIITILDE,
        gui.KeyName.DELETE: pyglet.window.key.DELETE,
        gui.KeyName.LEFT_SHIFT: pyglet.window.key.LSHIFT,
        gui.KeyName.RIGHT_SHIFT: pyglet.window.key.RSHIFT,
        gui.KeyName.LEFT_CONTROL: pyglet.window.key.LCTRL,
        gui.KeyName.RIGHT_CONTROL: pyglet.window.key.RCTRL,
        gui.KeyName.ALT: pyglet.window.key.MOD_ALT,
        gui.KeyName.META: pyglet.window.key.MOD_WINDOWS,
        gui.KeyName.CAPS_LOCK: pyglet.window.key.CAPSLOCK,
        gui.KeyName.LEFT: pyglet.window.key.LEFT,
        gui.KeyName.RIGHT: pyglet.window.key.RIGHT,
        gui.KeyName.UP: pyglet.window.key.UP,
        gui.KeyName.DOWN: pyglet.window.key.DOWN,
        gui.KeyName.INSERT: pyglet.window.key.INSERT,
        gui.KeyName.HOME: pyglet.window.key.HOME,
        gui.KeyName.END: pyglet.window.key.END,
        gui.KeyName.PAGE_UP: pyglet.window.key.PAGEUP,
        gui.KeyName.PAGE_DOWN: pyglet.window.key.PAGEDOWN,
        gui.KeyName.F1: pyglet.window.key.F1,
        gui.KeyName.F2: pyglet.window.key.F2,
        gui.KeyName.F3: pyglet.window.key.F3,
        gui.KeyName.F4: pyglet.window.key.F4,
        gui.KeyName.F5: pyglet.window.key.F5,
        gui.KeyName.F6: pyglet.window.key.F6,
        gui.KeyName.F7: pyglet.window.key.F7,
        gui.KeyName.F8: pyglet.window.key.F8,
        gui.KeyName.F9: pyglet.window.key.F9,
        gui.KeyName.F10: pyglet.window.key.F10,
        gui.KeyName.F11: pyglet.window.key.F11,
        gui.KeyName.F12: pyglet.window.key.F12
    }
    
    def _keyEventToFunction(self, keyEvent):
        if keyEvent.type == keyEvent.DOWN:
            if keyEvent.key == gui.KeyName.LEFT_SHIFT or keyEvent.key == gui.KeyName.RIGHT_SHIFT:
                self._modifiers |= gui.KeyModifier.SHIFT.value
            if keyEvent.key == gui.KeyName.LEFT_CONTROL or keyEvent.key == gui.KeyName.RIGHT_CONTROL:
                self._modifiers |= gui.KeyModifier.CTRL.value
            if keyEvent.key == gui.KeyName.ALT:
                self._modifiers |= gui.KeyModifier.ALT.value
            if keyEvent.key == gui.KeyName.META:
                self._modifiers |= gui.KeyModifier.META.value
            if keyEvent.key in self._key_to_symbol:
                self.on_key_press(self._key_to_symbol[keyEvent.key], self._modifiers)
                return True
        elif keyEvent.type == keyEvent.UP:
            if keyEvent.key == gui.KeyName.LEFT_SHIFT or keyEvent.key == gui.KeyName.RIGHT_SHIFT:
                self._modifiers &= ~gui.KeyModifier.SHIFT.value
            if keyEvent.key == gui.KeyName.LEFT_CONTROL or keyEvent.key == gui.KeyName.RIGHT_CONTROL:
                self._modifiers &= ~gui.KeyModifier.CTRL.value
            if keyEvent.key == gui.KeyName.ALT:
                self._modifiers &= ~gui.KeyModifier.ALT.value
            if keyEvent.key == gui.KeyName.META:
                self._modifiers &= ~gui.KeyModifier.META.value
            if keyEvent.key in self._key_to_symbol:
                self.on_key_release(self._key_to_symbol[keyEvent.key], self._modifiers)
                return True
        else:
            raise NotImplementedError("KeyEvent is neither of type UP nor DOWN")
        
    def _mouseEventToFunction(self, mouseEvent):
        if mouseEvent.type in (gui.MouseEvent.BUTTON_DOWN, gui.MouseEvent.DRAG, gui.MouseEvent.BUTTON_UP):
            screen_x = mouseEvent.x - self._scene_widget.frame.x
            screen_y = mouseEvent.y - self._scene_widget.frame.y

            if screen_x < 0:
                screen_x = 0
            elif screen_x > self._scene_widget.frame.width - 1:
                screen_x = self._scene_widget.frame.width - 1

            if screen_y < 0:
                screen_y = 0
            elif screen_y > self._scene_widget.frame.height - 1:
                screen_y = self._scene_widget.frame.height - 1

            button = 0
            if mouseEvent.buttons & gui.MouseButton.LEFT.value:
                button = pyglet.window.mouse.LEFT
            elif mouseEvent.buttons & gui.MouseButton.RIGHT.value:
                button = pyglet.window.mouse.RIGHT
            elif mouseEvent.buttons & gui.MouseButton.MIDDLE.value:
                button = pyglet.window.mouse.MIDDLE
            elif mouseEvent.buttons & gui.MouseButton.BUTTON4.value:
                button = pyglet.window.mouse.MOUSE4
            elif mouseEvent.buttons & gui.MouseButton.BUTTON5.value:
                button = pyglet.window.mouse.MOUSE5
            
            if mouseEvent.type == gui.MouseEvent.BUTTON_DOWN:
                self.on_mouse_press(screen_x, screen_y, -np.inf, button, self._modifiers)
            elif mouseEvent.type == gui.MouseEvent.DRAG:                    
                self.on_mouse_drag(screen_x, screen_y, -np.inf, screen_x - self._last_coords[0], screen_y - self._last_coords[1], -np.inf, mouseEvent.buttons, self._modifiers)
            elif mouseEvent.type == gui.MouseEvent.BUTTON_UP:
                self.on_mouse_release(screen_x, screen_y, -np.inf, button, self._modifiers)
            else:
                # Unsupported mouse type; do nothing
                pass

            self._last_coords = (screen_x, screen_y)

        else:
                # Unsupported mouse type; do nothing
                pass

        return gui.Widget.EventCallbackResult.HANDLED
         

    def addShape(self, shape:Shape|ShapeSet, name:None|str=None, quick:bool=False):
        '''Adds a shape to the scene.

        Args:
            shape: The shape to add.
            name: Name of the shape.
            quick: If this method is called in rapid succession e.g., inside
                Scene3d.on_idle, set quick=True, which might prevent some
                crashes.
        '''
        # check type of shape
        if quick:
            gui.Application.instance.post_to_main_thread(self._window, lambda: shape._addToScene(self, name))
        else:
            shape._addToScene(self, name)
        

    def updateShape(self, name:str, quick:bool=False):
        '''Update a shape in the scene.

        Args:
            name: Name of the shape to update.
            quick: If this method is called in rapid succession e.g., inside
                Scene3d.on_idle, set quick=True, which might prevent some
                crashes.
        '''
        if self._scene_widget.scene.has_geometry(name):
            if quick:
                # The documentation recommends this instead, but it seems to only update after another action
                # has been taken (moving the mouse, pressing a key, etc.)
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._shapeDict[name]._update(name, self))
            else:
                self._shapeDict[name]._update(name, self)
        elif name in self._shapeDict:
            shape = self._shapeDict[name]
            self.removeShape(name)
            self.addShape(shape, name, quick)



    def removeShape(self, name:str):
        '''Removes a shape from the scene.

        Args:
            name: Name of the shape to remove.
        '''
        if self._scene_widget.scene.has_geometry(name):
            self._scene_widget.scene.remove_geometry(name)
        elif name in self._shapeDict and isinstance(self._shapeDict[name]._shape, gui.Label3D):
            self._scene_widget.remove_3d_label(self._shapeDict[name]._shape)
        if name in self._shapeDict:
            del self._shapeDict[name]

    def print(self, *args, **kwargs):
        '''Prints text to the scene's text output.

        Prints text to the scene's text output, on the left side of the
        window. Supports the same arguments as the built-in print function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        '''
        print(*args, **kwargs, file=self._sio)
        self._text_output.text = self._sio.getvalue()[-3072:]

    def set_slider_value(self, slider_id:int, value:Number, no_callback:bool=False):
        '''Sets the value of a slider, programmatically.

        Args:
            slider_id: The id of the slider to set.
            value: The value to set the slider to.
            no_callback: If True, Scene3D.on_slider_change will not be
                triggered.
        '''
        if slider_id > len(self._sliders):
            raise IndexError("slider_id too large!")
        
        self._sliders[slider_id].double_value = value
        if not no_callback:
            self.on_slider_change(slider_id, value)

    # def show_axes(self, enable=True):
    #     '''
    #     It doesn't work!
    #     '''
    #     self._scene_widget.scene.show_axes(enable)
        
        
def get_rotation_matrix(angle:float, axis:NDArray3|List3|Tuple3) -> np.ndarray:
    '''Returns a rotation matrix from its axis-angle representation.

    Args:
        angle (float): Rotation angle.
        axis (np.ndarray, list, tuple): Rotation axis.

    Returns:
        The respective rotation matrix.
    '''
    if isinstance(axis, (np.ndarray, list, tuple)):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * axis)
    else:
        raise TypeError("Incorrect type for axis")
        
def world_space(func:Callable) -> Callable:
    '''Decorator to convert screen coordinates to world coordinates.

    Use this decorator to convert screen-space coordinates to world-space
    coordinates in Scene3D.on_mouse_press, Scene3D.on_mouse_drag, and
    Scene3D.on_mouse_release. In that case, x, y, and z will all correspond
    to the world-space coordinates of a projected ray from the mouse.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    '''
    argspec = getfullargspec(func)
    def wrapper(*args, **kwargs):
        try:
            scene = args[argspec.args.index("self")]
        except IndexError:
            scene = kwargs["self"]

        if "x" in argspec.args and "y" in argspec.args and "z" in argspec.args:
            try:
                x = args[argspec.args.index("x")]
            except IndexError:
                x = kwargs["x"]
            try:
                y = args[argspec.args.index("y")]
            except IndexError:
                y = kwargs["y"]

            def screen_to_world(depth_image):
                depth = np.asarray(depth_image)[y, x]
                world = scene._scene_widget.scene.camera.unproject(x, y, depth, scene._scene_widget.frame.width, scene._scene_widget.frame.height)

                if "dx" in argspec.args and "dy" in argspec.args and "dz" in argspec.args:
                    try:
                        dx = args[argspec.args.index("dx")]
                    except IndexError:
                        dx = kwargs["dx"]
                    try:
                        dy = args[argspec.args.index("dy")]
                    except IndexError:
                        dy = kwargs["dy"]

                    ddepth = np.asarray(depth_image)[y - dy, x - dx]
                    dworld = scene._scene_widget.scene.camera.unproject(x - dx, y - dy, ddepth, scene._scene_widget.frame.width, scene._scene_widget.frame.height)

                new_kwargs = {}
                for arg in argspec.args:
                    if arg == "x":
                        new_kwargs["x"] = world[0]
                    elif arg == "y":
                        new_kwargs["y"] = world[1]
                    elif arg == "z":
                        new_kwargs["z"] = world[2]
                    elif arg == "dx":
                        new_kwargs["dx"] = world[0] - dworld[0] if not np.isinf(world[0]) and not np.isinf(dworld[0]) else 0
                    elif arg == "dy":
                        new_kwargs["dy"] = world[1] - dworld[1] if not np.isinf(world[1]) and not np.isinf(dworld[1]) else 0
                    elif arg == "dz":
                        new_kwargs["dz"] = world[2] - dworld[2] if not np.isinf(world[2]) and not np.isinf(dworld[2]) else 0
                    else:
                        try:
                            value = args[argspec.args.index(arg)]
                        except IndexError:
                            value = kwargs[arg]
                        finally:
                            new_kwargs[arg] = value

                func(**new_kwargs)

            scene._scene_widget.scene.scene.render_to_depth_image(screen_to_world)

    return wrapper
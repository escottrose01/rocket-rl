"""A simple 2D game engine."""

import numpy as np
import pygame
from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

TITLE = 'RL-Rocket'
WIDTH = 800
HEIGHT = 600
FPS = 60
SCALE = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CONSISTENT_PHYSICS = True

pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])

game_instance = None

# forward declaration of classes


class Game:
    pass


class Collision:
    pass


class GameObject(object):
    """An abstract game object type. Subclases should override the get_sprite method to facilitate rendering."""

    def __init__(self, game: Game = None):
        self._game = game or Game.instance()
        self._game.add_object(self)

    def destroy(self):
        self._game.remove_object(self)

    def update(self, dt: float):
        pass

    def get_sprites(self) -> pygame.sprite.Sprite:
        return tuple()


class Camera(GameObject):
    """A camera class to facilitate rendering pygame simulations."""

    # TODO(escottrose01): implement rotations

    def __init__(self, x: float = WIDTH//2, y: float = HEIGHT//2, rotation: float = 0, width: float = 800, height: float = 600):
        """Initializes a new Camera instance.

        Args:
            x (float, optional): The x-coordinate of the camera center. Defaults to 0.
            y (float, optional): The y-coordinate of the camera center. Defaults to 0.
            rotation (float, optional): The rotation of the camera. Defaults to 0.
            width (float, optional): the width of the camera viewport in world coordinates. Defaults to 400.
            height (float, optional): the height of the camera viewport in world coordinates. Defaults to 300.
        """
        super().__init__()

        self.x = x
        self.y = y
        self.rotation = rotation
        self.width = width
        self.height = height
        self.scalex = WIDTH / self.width
        self.scaley = HEIGHT / self.height

        self._game._camera = self

    def transform(self, sprite: pygame.sprite.Sprite, rect: pygame.rect.Rect):
        """Transforms an input sprite according to the camera parameters.

        Args:
            sprite (pygame.sprite.Sprite): The sprite to transform.
            rect (pygame.rect.Rect): The rect of the sprite.
        """
        w, h = rect.w, rect.h
        x, y = rect.center
        res_rect = pygame.rect.Rect(0, 0, w, h)
        res_rect.center = (WIDTH//2 - self.scalex * (self.x - x), HEIGHT//2 - self.scaley * (-self.y + y))
        sprite = pygame.transform.scale(sprite, (w*self.scalex, h*self.scaley))
        res_rect = sprite.get_rect(center=res_rect.center)
        return sprite, res_rect


class Game(object):
    """Manages the rocket game engine."""

    def __init__(self, callbacks: list = None):
        """Initializes a new Game instance.

        Args:
            callbacks (list, optional): the list of functions to call after each game loop. Defaults to [].
        """
        global game_instance
        if game_instance is not None:
            raise Exception('Error: physics instance already exists')

        self._callbacks = callbacks or []
        self._graphics = True
        self._run = True
        self._objects = []
        self._camera = None

    def instance():
        global game_instance
        if game_instance is None:
            game_instance = Game()
        return game_instance

    @ property
    def graphics(self) -> bool:
        """Returns the current graphics setting of this game.

        Returns:
            bool: whether or not this game renders updates.
        """
        return self._graphics

    @ graphics.setter
    def graphics(self, value: bool):
        """Sets the graphics setting of this game.

        Args:
            value (bool): whether or not this game should render updates.
        """
        self._graphics = value

    def register_callback(self, f):
        """Registers a callback to this game instance.

        Args:
            f (function): the callback.
        """
        self._callbacks.append(f)

    def add_object(self, obj: GameObject):
        self._objects.append(obj)

    def delete_object(self, obj: GameObject):
        self._objects.remove(obj)

    def render(self):
        screen.fill(BLACK)
        for obj in self._objects:
            for entity in obj.get_sprites():
                if self._camera and not entity.ui:
                    screen.blit(*self._camera.transform(entity.image, entity.rect))
                else:
                    screen.blit(entity.image, entity.rect)
        pygame.display.flip()

    def end(self):
        """Ends the game, but does not close the pygame window."""
        self._run = False

    def reset(self):
        self._objects = []
        self._callbacks = []
        Physics.reset()

    def run(self):
        """Runs the simulation until the user closes out."""

        clock = pygame.time.Clock()

        while self._run:
            # simulation update
            if self._graphics:
                dt = SCALE * clock.tick(FPS) / 1000
            else:
                dt = dt = SCALE / FPS

            if CONSISTENT_PHYSICS:
                dt = dt = SCALE / FPS

            for obj in self._objects:
                obj.update(dt)

            Physics.instance().step(dt)

            # graphics
            if self._graphics:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            return
                    elif event.type == QUIT:
                        return

                self.render()

            # callbacks
            for f in self._callbacks:
                f(self, dt)

    def step(self, dt):
        """Steps forward one timestep in the game loop"""

        for obj in self._objects:
            obj.update(dt)

        Physics.instance().step(dt)

        # graphics
        if self._graphics:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return
                elif event.type == QUIT:
                    return

            self.render()

        # callbacks
        for f in self._callbacks:
            f(self, dt)

        # clear event queue
        pygame.event.get()


class RigidBody(GameObject):
    """An abstract rigid body."""

    # TODO(escottrose01): Add generic rigidbody collision detection and resolution.

    def __init__(self, mass: float, moment_of_inertia: float, position: list, rotation: float, width: float = 0.0, height: float = 0.0):
        """Initializes a new RigidBody instance.

        Args:
            mass (float): the mass of this body.
            moment_of_inertia (float): the moment of inertia of this body.
            position (array_like): the initial (x, y) position of this body.
            rotation (float): the heading of this body, in radians.
            width (float): the width of the collider on this body.
            height (float): the height of the collider on this body.
        """
        super().__init__()

        # physical properties
        self._m = mass
        self._mi = moment_of_inertia
        self._p = np.array(position, dtype=np.float64)
        self._v = np.array((0, 0), dtype=np.float64)
        self._r = rotation
        self._av = 0

        self._f = np.array((0, 0), dtype=np.float64)
        self._t = 0

        # collision properties
        self._w = width
        self._h = height

        Physics.instance().add_body(self)

    @ property
    def position(self) -> np.ndarray:
        """Returns the position of this body in world coordinates.

        Returns:
            ndarray: the (x, y) position of this body.
        """
        return self._p

    @ position.setter
    def position(self, value: list):
        """Sets the position of this body

        Args:
            value (array_like): The velocity of this body, in world coordinates.
        """
        self._p = np.array(value)

    @ property
    def rotation(self) -> float:
        """Returns the rotation of this body in world coordinates.

        Returns:
            float: the rotation of this body (radians).
        """
        return self._r

    @ property
    def mass(self) -> float:
        """Returns the mass of this body.

        Returns:
            float: the mass of this body.
        """
        return self._m

    @ property
    def velocity(self) -> np.ndarray:
        """Returns the velocity of this body.

        Returns:
            ndarray: the velocity of this body.
        """
        return self._v

    @ velocity.setter
    def velocity(self, value: list):
        """Sets the velocity of this body.

        Args:
            value (array_like): the velocity of this body.
        """
        self._v = np.array(value)

    @ property
    def angular_velocity(self) -> float:
        """Returns the angular velocity of this body.

        Returns:
            float: the angular velocity of this body.
        """
        return self._av

    @ angular_velocity.setter
    def angular_velocity(self, value: float):
        """Sets the angular velocity of this body.

        Args:
            value (float): the angular velocity of this body.
        """
        self._av = value

    @ property
    def collider(self) -> list:
        """Returns the rectangle collider of this body, as a list of four points.

        Returns:
            list: the collision rectangle of this body.
        """
        m = np.array([[np.cos(self._r), -np.sin(self._r)], [np.sin(self._r), np.cos(self._r)]])
        # print(list(self._p[:, np.newaxis] + m.dot([[-self._w/2, -self._w/2, self._w/2, self._w/2],
        #   [-self._h/2, self._h/2, -self._h/2, self._h/2]])))
        # return (self._p[:, np.newaxis] + m.dot([[-self._w/2, -self._w/2, self._w/2, self._w/2], [-self._h/2, self._h/2, -self._h/2, self._h/2]])).T.tolist()
        return (self._p[:, np.newaxis] + m.dot([[-self._w/2, -self._w/2, self._w/2, self._w/2], [0.0, -self._h, 0.0, -self._h]])).T.tolist()
        # return (self._p[0] - self._w/2, self._p[1] - self._h/2, self._w, self._h)

    def add_force(self, force: list, contact_point: list = None):
        """Applies a force to this RigidBody.

        Args:
            force (array_like): force vector in world coordinates.
            contact_point (array_like, optional): if set, the point at which to apply the force. Defaults to None.
        """
        cp = contact_point or self._p
        self._f += force
        self._t += np.cross(self._p - cp, force)

    def add_torque(self, torque: float):
        """Applies a torque to this RigidBody.

        Args:
            torque (float): the torque to apply.
        """
        self._t += torque

    def on_collision(self, collision: Collision):
        """Called when two rigidbodies collides

        Args:
            collision (Collision): the collision with this RigidBody.
        """
        pass

    def fixed_update(self, dt: float):
        """Called every simulation timestep.

        Args:
            dt (float): the time since the last timestep.
        """
        pass

    def step(self, dt: float):
        """Applies pending forces to this RigidBody to update position and velocity.

        Args:
            dt (float): the time since the last timestep.
        """
        # Linear component
        a = self._f / self._m
        self._v += a * dt
        self._p += self._v * dt + 0.5 * a * dt * dt

        # Angular component
        aa = self._t / self._mi
        self._av += aa * dt
        self._r += self._av * dt + 0.5 * aa * dt * dt
        self._r = (self._r - np.pi) % (2*np.pi) - np.pi

        # Clear forces
        self._f = np.array((0, 0), dtype=np.float64)
        self._t = 0


class UpdateListener(object):
    """An abstract class that responds to physics updates. These objects are updated before RigidBodies."""

    def __init__(self):
        """Initializes a new UpdateListener instance.
        """
        Physics.instance().add_listener(self)

    def on_fixed_update(self, dt):
        """Called once every simulation timestep.

        Args:
            dt (float): the time since the last timestep.
        """
        pass


class Collision(object):
    """A class to hold information related to a collision"""

    def __init__(self, body: RigidBody, position: list, normal: list, velocity: list, bounce: float = 0.0, friction: float = 0.0):
        """Initialize a new Collision instance.

        Args:
            body (RigidBody): the body colliding.
            position (array_like): the position to which to snap the colliding body.
            normal (array_like): the normal vector along which the collision is taking place.
            velocity (array_like): the current velocity of the colliding body.
            bounce (float): the amount of bounce to apply in the opposite direction.
        """
        self.body = body
        self.position = position
        self.normal = normal
        self.velocity = velocity
        self.bounce = bounce
        self.friction = friction
        self.t = np.inner(self.velocity, self.normal)


# physics_instance = None


class Physics(object):
    """A simulation to keep track of several bodies and their interactions"""

    physics_instance = None

    def instance():
        if Physics.physics_instance is None:
            Physics.physics_instance = Physics()
        return Physics.physics_instance

    def reset():
        Physics.physics_instance = None

    def __init__(self, bodies: list = None, listeners: list = None):
        """Initializes a new Physics instance.

        Args:
            bodies (list_like, optional): the list of bodies to simulate. Defaults to [].
            listeners (list_like, optional): the list of physics update listeners. Defaults to [].

        Raises:
            Exception: The Physics class follows the singleton pattern, and only a single instance may be created.
        """
        if Physics.physics_instance is not None:
            raise Exception('Error: physics instance already exists')
        self._bodies = bodies or []
        self._listeners = listeners or []
        self._collisions = []

    @ property
    def bodies(self) -> list:
        """Returns the list of bodies being simulated.

        Returns:
            list_like: The list of bodies.
        """
        return self._bodies

    def add_body(self, body: RigidBody):
        """Adds a body to the simulation.

        Args:
            body (RigidBody): the body to add to the simulation.
        """
        self._bodies.append(body)

    def add_listener(self, listener: UpdateListener):
        """Adds a listener to the simulation.

        Args:
            listener (UpdateListener): the listener to add to the simulation.
        """
        self._listeners.append(listener)

    def add_collision(self, collision: Collision):
        """Registers a collision to be handled in the next timestep.
        This simple physics simulation uses inelastic collision resolution.

        Args:
            collision (Collision): the collison to handle.
        """
        self._collisions.append(collision)

    def step(self, dt: float):
        """Moves forward one timestep in the simulation.

        Args:
            dt (float): The elapsed time (ms) since the previous timestep.
        """
        for l in self._listeners:
            l.on_fixed_update(dt)

        for b in self._bodies:
            b.fixed_update(dt)

        for c in self._collisions:
            c.body.on_collision(c, dt)

        self._collisions = []

        for b in self._bodies:
            b.step(dt)

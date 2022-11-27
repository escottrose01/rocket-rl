"""A rigid body physics simulator."""

import numpy as np


class RigidBody(object):
    """An abstract rigid body."""

    def __init__(self, mass: float, moment_of_inertia: float, position: list, rotation: float):
        """Initializes a new RigidBody instance.

        Args:
            mass (float): the mass of this body.
            moment_of_inertia (float): the moment of inertia of this body.
            position (array_like): the initial (x, y) position of this body.
            rotation (float): the heading of this body, in radians.
        """
        self._m = mass
        self._mi = moment_of_inertia
        self._p = np.array(position, dtype=np.float64)
        self._v = np.array((0, 0), dtype=np.float64)
        self._r = rotation
        self._av = 0

        self._f = np.array((0, 0), dtype=np.float64)
        self._t = 0

        Physics.instance().add_body(self)

    @property
    def position(self) -> np.ndarray:
        """Returns the position of this body in world coordinates.

        Returns:
            ndarray: the (x, y) position of this body.
        """
        return self._p

    @position.setter
    def position(self, value: list):
        """Sets the position of this body

        Args:
            value (array_like): The velocity of this body, in world coordinates.
        """
        self._p = np.array(value)

    @property
    def rotation(self) -> float:
        """Returns the rotation of this body in world coordinates.

        Returns:
            float: the rotation of this body (radians).
        """
        return self._r

    @property
    def mass(self) -> float:
        """Returns the mass of this body.

        Returns:
            float: the mass of this body.
        """
        return self._m

    @property
    def velocity(self) -> np.ndarray:
        """Returns the velocity of this body.

        Returns:
            ndarray: the velocity of this body.
        """
        return self._v

    @velocity.setter
    def velocity(self, value: list):
        """Sets the velocity of this body.

        Args:
            value (array_like): the velocity of this body.
        """
        self._v = np.array(value)

    @property
    def angular_velocity(self) -> float:
        """Returns the angular velocity of this body.

        Returns:
            float: the angular velocity of this body.
        """
        return self._av

    @angular_velocity.setter
    def angular_velocity(self, value: float):
        """Sets the angular velocity of this body.

        Args:
            value (float): the angular velocity of this body.
        """
        self._av = value

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

    def update(self, dt: float):
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
        self._p += self._v * dt

        # Angular component
        aa = self._t / self._mi
        self._av += aa * dt
        self._r += self._av * dt

        # Clear forces
        self._f = np.array((0, 0), dtype=np.float64)
        self._t = 0


class UpdateListener(object):
    """An abstract class that responds to physics updates. These objects are updated before RigidBodies."""

    def __init__(self):
        """Initializes a new UpdateListener instance.
        """
        Physics.instance().add_listener(self)

    def update(self, dt):
        """Called once every simulation timestep.

        Args:
            dt (float): the time since the last timestep.
        """
        pass


physics_instance = None


class Physics(object):
    """A simulation to keep track of several bodies and their interactions"""

    # TODO(escottrose01): Capture collisions inside a class, not tuples
    # TODO(escottrose01): Add better friction on ground collisions

    def instance():
        global physics_instance
        if physics_instance is None:
            physics_instance = Physics()
        return physics_instance

    def __init__(self, bodies: list = [], listeners: list = []):
        """Initializes a new Physics instance.

        Args:
            bodies (list_like, optional): the list of bodies to simulate. Defaults to [].
            listeners (list_like, optional): the list of physics update listeners. Defaults to [].

        Raises:
            Exception: The Physics class follows the singleton pattern, and only a single instance may be created.
        """
        if physics_instance is not None:
            raise Exception('Error: physics instance already exists')
        self._bodies = bodies
        self._listeners = listeners
        self._collisions = []

    @property
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

    def add_collision(self, collision: list):
        """Registers a collision to be handled in the next timestep.
        This simple physics simulation uses inelastic collision resolution.

        Args:
            collision (list_like): a (body, position, direction) tuple to resolve.
            The body's position and velocity are snapped according to the provided
            position and direction.
        """
        self._collisions.append(collision)

    def step(self, dt: float):
        """Moves forward one timestep in the simulation.

        Args:
            dt (float): The elapsed time (ms) since the previous timestep.
        """
        for l in self._listeners:
            l.update(dt)

        for b in self._bodies:
            b.update(dt)

        for c in self._collisions:
            body = c[0]
            position = c[1]
            normal = c[2]

            body.position = position
            s = np.inner(body.velocity, normal)
            v_n = max(s, 0.0) * normal
            # v_t = body.velocity - s * normal
            body.velocity = v_n  # + v_t
            body.angular_velocity = 0.0
        self._collisions = []

        for b in self._bodies:
            b.step(dt)

import pygame
from pygame.locals import K_w, K_a, K_s, K_d, K_q, K_e

from engine import Camera, RigidBody, WIDTH, HEIGHT


# TODO(escottrose01): Implement follow cameras for GameObjects (add transform to GameObject!)
class FollowCamera(Camera):
    def __init__(self, target: RigidBody, rotation: float = 0, width: float = 400, height: float = 300):
        super().__init__(target.position[0], target.position[1], rotation, width, height)
        self.target = target

    def transform(self, sprite: pygame.sprite.Sprite, rect: pygame.rect.Rect):
        return super().transform(sprite, rect)

    def update(self, dt):
        self.x = self.target.position[0]
        self.y = self.target.position[1]


class KeyboardCamera(Camera):
    def __init__(self, x: float = WIDTH//2, y: float = HEIGHT//2, rotation: float = 0, width: float = 800, height: float = 600):
        super().__init__(x, y, rotation, width, height)
        self._dir = (0, 0)
        self.speed = 100.0
        self._width = width
        self._height = self.height
        self.scale = 1.0
        self.zoom = 0.0

    def update(self, dt: float):
        self.x += self.speed * self._dir[0] * dt
        self.y += self.speed * self._dir[1] * dt
        self.scale *= (1.0 + dt * 0.1 * self.zoom)

        self.height = self._height * self.scale
        self.width = self._width * self.scale
        self.scalex = WIDTH / self.width
        self.scaley = HEIGHT / self.height

    def accept_input(self, pressed_keys):
        dir = [0, 0]
        self.zoom = 0
        if pressed_keys[K_w]:
            dir[1] = 1
        if pressed_keys[K_a]:
            dir[0] = -1
        if pressed_keys[K_s]:
            dir[1] = -1
        if pressed_keys[K_d]:
            dir[0] = 1
        if pressed_keys[K_e]:
            self.zoom = 1
        if pressed_keys[K_q]:
            self.zoom = -1
        self._dir = tuple(dir)

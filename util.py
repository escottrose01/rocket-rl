import pygame

import keras

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
# from rl.memory import SequentialMemory

from engine import GameObject

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TRANS = (0, 0, 0, 0)

font = pygame.font.Font(None, 24)


class Circle(GameObject):
    def __init__(self, x, y, r, color=(150, 25, 25), ui=False):
        super().__init__()
        self.sprite = Circle.Sprite(x, y, r, color, ui)

    def get_sprites(self) -> pygame.sprite.Sprite:
        return self.sprite,

    class Sprite(pygame.sprite.Sprite):
        def __init__(self, x, y, r, color=(150, 25, 25), ui=False):
            super().__init__()
            self.ui = ui
            self.image = pygame.Surface((2*r, 2*r), pygame.SRCALPHA)
            self.image.fill(TRANS)
            pygame.draw.circle(self.image, color, (r, r), r, 0)
            self.rect = pygame.Rect(x - r, y - r, self.image.get_width(), self.image.get_height())


class Text(GameObject):
    def __init__(self, x, y, text='', color=WHITE, bgcolor=TRANS):
        super().__init__()
        self._color = color
        self._bgcolor = bgcolor
        self._x = x
        self._y = y
        self.sprite = Text.Sprite(x, y, text, color, bgcolor)

    def get_sprites(self) -> pygame.sprite.Sprite:
        return self.sprite,

    def set_text(self, text: str):
        self.sprite.image = font.render(text, True, self._color, self._bgcolor)
        self.sprite.rect = self.sprite.image.get_rect()
        self.sprite.rect.topleft = (self._x, self._y)

    class Sprite(pygame.sprite.Sprite):
        def __init__(self, x, y, text, color, bgcolor):
            super().__init__()
            self.ui = True
            self.image = font.render(text, True, color, bgcolor)
            self.rect = self.image.get_rect()
            self.rect.topleft = (x, y)

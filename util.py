import pygame


class Circle(pygame.sprite.Sprite):
    def __init__(self, x, y, r, color=(150, 25, 25)):
        super().__init__()
        self.image = pygame.Surface((2*r, 2*r))
        pygame.draw.circle(self.image, color, (r, r), r, 0)
        self.rect = pygame.Rect(x - r, y - r, self.image.get_width(), self.image.get_height())

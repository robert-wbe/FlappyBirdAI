import sys
import pygame
import random
import neuralnetwork as nn

pygame.init()

SCREEN_WIDTH = 576
SCREEN_HEIGHT = 1024

class Pipe:
    BUFFER = 150
    DISTANCE = 250
    img = pygame.image.load("assets/pipe-green.png")
    lower = pygame.transform.scale2x(img)
    upper = pygame.transform.flip(lower, False, True)
    def __init__(self) -> None:
        self.height = random.randint(self.BUFFER, SCREEN_HEIGHT-150-self.BUFFER)
        self.lowerRect = self.lower.get_rect()
        self.lowerRect.topleft = (SCREEN_WIDTH, self.height+self.DISTANCE//2)
        self.upperRect = self.upper.get_rect()
        self.upperRect.bottomleft = (SCREEN_WIDTH, self.height-self.DISTANCE//2)
    
    def update(self):
        self.lowerRect.centerx -= 3
        self.upperRect.centerx -= 3

class Bird:
    GRAVITY = 0.5
    JUMP_STRENGTH = 12

    def __init__(self, initialY=SCREEN_HEIGHT//2, isYellow=False) -> None:
        self.isActive = True
        self.yVelocity = 0
        img = pygame.image.load("assets/yellowbird-midflap.png").convert_alpha() if isYellow else pygame.image.load("assets/bluebird-midflap.png").convert_alpha()
        self.img = pygame.transform.scale2x(img)
        self.rect = self.img.get_rect(center=(100, initialY))
        self.network = nn.NeuralNetwork(5, 5, 2)
    
    def checkCollision(self, pipes: list[Pipe]):
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT-100:
            return True
        for pipe in pipes:
            if self.rect.colliderect(pipe.lowerRect) or self.rect.colliderect(pipe.upperRect):
                return True
    
    def jump(self):
        self.yVelocity = -self.JUMP_STRENGTH
    
    def update(self):
        self.yVelocity += self.GRAVITY
        self.rect.centery += self.yVelocity
    
    def think(self, nextPipe: Pipe):
        #input order: [yPos, yVel, distance, upper, lower]
        results = self.network.run([self.rect.centery, self.yVelocity, nextPipe.lowerRect.centerx, nextPipe.upperRect.bottom, nextPipe.lowerRect.top])
        #output order: [not jump, jump]
        if results[1] > results[0]:
            self.jump()
        
    def mutate(self):
        new = Bird()
        new.network = self.network.mutate(random.random()*0.5)
        return new
    

class FlappyBirdAIGame:
    FPS = 60
    GENERATION_SIZE = 10
    SPAWNPIPE = pygame.USEREVENT
    FLAPPY_BIRD_FONT = pygame.font.SysFont("Impact", 32)
    SIM_SPEEDS = (1,4,16)
    speedRects = (pygame.Rect(320, 60, 65, 32), pygame.Rect(405, 60, 65, 32), pygame.Rect(490, 60, 65, 32))
    speedImgs = (pygame.image.load("assets/normal-speed.png"), pygame.image.load("assets/medium-speed.png"), pygame.image.load("assets/full-speed.png"))

    def __init__(self) -> None:
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.simulationSpeed = 1
        self.gameTime = 1
        self.groundX = 0
        self.highScore = 0
        self.generation = 1

        background = pygame.image.load("assets/background-day.png").convert()
        self.background = pygame.transform.scale2x(background)
        ground = pygame.image.load("assets/base.png").convert()
        self.ground = pygame.transform.scale2x(ground)
        pipe = pygame.image.load("assets/pipe-green.png").convert()
        self.pipe = pygame.transform.scale2x(pipe)

        self.pipes: list[Pipe] = [Pipe()]
        self.birds: list[Bird] = [Bird() for _ in range(self.GENERATION_SIZE)]
        self.bestBird = self.birds[0]
    
    def newGen(self):
        self.birds = [self.bestBird.mutate() for _ in range(self.GENERATION_SIZE)]
        self.pipes = [Pipe()]
        self.gameTime = 1
        self.generation += 1

    def draw(self):
        self.screen.blit(self.background, (0,0))
        for pipe in self.pipes:
            self.screen.blit(pipe.lower, pipe.lowerRect)
            self.screen.blit(pipe.upper, pipe.upperRect)
        self.screen.blit(self.ground, (-self.groundX,900))
        for bird in self.birds:
            if bird.isActive:
                self.screen.blit(bird.img, bird.rect)
        timeAlive = self.FLAPPY_BIRD_FONT.render(f"TIME ALIVE: {self.gameTime//self.FPS}", True, (0,0,0))
        highScore = self.FLAPPY_BIRD_FONT.render(f"BEST AI: {self.highScore//self.FPS}", True, (0,0,0))
        speedChooser = self.FLAPPY_BIRD_FONT.render(f"SIMULATION SPEED", True, (0,0,0))
        genCounter = self.FLAPPY_BIRD_FONT.render(f"GEN {self.generation}", True, (0,0,0))
        self.screen.blit(timeAlive, (20, 20))
        self.screen.blit(highScore, (20, 60))
        self.screen.blit(speedChooser, (320, 20))
        self.screen.blit(genCounter, (215, 20))

        for i in range(3):
            self.screen.blit(self.speedImgs[i], self.speedRects[i])

        pygame.display.update()
    
    def update(self):
        self.gameTime += 1
        if self.gameTime > self.highScore:
            self.highScore = self.gameTime
        stop = True
        self.groundX = ((self.groundX + 3) % 48)
        if self.gameTime % 120 == 0:
            if len(self.pipes) >= 3:
                self.pipes.pop(0)
            self.pipes.append(Pipe())
        for pipe in self.pipes:
            pipe.update()
        for bird in self.birds:
            if bird.isActive:
                if bird.checkCollision(self.pipes):
                    bird.isActive = False
                    self.bestBird = bird
                    continue
                stop = False
                nextPipe = self.pipes[-2] if len(self.pipes) > 1 and self.pipes[-2].lowerRect.right >= bird.rect.left else self.pipes[-1]
                bird.think(nextPipe)
                bird.update()
        if stop:
            self.newGen()
    
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for i in range(3):
                        if self.speedRects[i].collidepoint(pos):
                            self.simulationSpeed = self.SIM_SPEEDS[i]
                            break
                # if event.type == pygame.KEYDOWN:
                #     if event.key == pygame.K_SPACE:
                #         pass
            for _ in range(self.simulationSpeed):
                self.update()
            self.draw()
            self.clock.tick(self.FPS)
    

if __name__ == "__main__":
    FlappyBirdAIGame().run()
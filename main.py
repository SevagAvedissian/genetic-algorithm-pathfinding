import pygame
import numpy as np
import sys
import random
import math
import operator
from config import *

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


class Obstacles:
    def __init__(self):
        self.list = []

    def add(self, x, y, w, h):
        obstacle = pygame.Rect(0, 0, w, h)
        obstacle.center = (x, y)
        self.list.append(obstacle)

    def draw(self):
        for obstacle in self.list:
            pygame.draw.rect(display, BLACK, obstacle)


obstacles = Obstacles()
obstacles.add(WIDTH/2, 500, 500, 10)
obstacles.add(200, 250, 450, 10)
obstacles.add(WIDTH-200, 250, 450, 10)
obstacles.add(WIDTH/2, 140, 200, 10)

# different set of obstacles
#obstacles.add(WIDTH / 2, 550, 400, 10)
#obstacles.add(300, 595, 10, 100)
#obstacles.add(WIDTH - 300, 595, 10, 100)

pygame.init()

display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Genetic Algorithm")
arial = pygame.font.SysFont("arial", 30)


class Individual:
    def __init__(self, x, y, DNA=None):
        if DNA is None:
            DNA = []
        self.pos = np.array([x, y])
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        self.DNA = DNA
        self.size = dot_size
        self.color = WHITE
        self.counter = 0
        self.fitness = 0
        self.dead = False
        self.win = False

    def draw(self):
        pygame.draw.circle(display, self.color, self.pos, self.size)

    def generate_DNA(self):
        for i in range(num_of_moves):
            self.DNA.append(
                np.array([random.randint(-3, 3), random.randint(-3, 3)]))

    def update(self):
        if not self.dead and not self.win:
            try:
                self.acc = self.DNA[self.counter]
            except IndexError:
                self.dead = True
            self.vel = np.add(self.acc, self.vel)
            self.vel = np.clip(self.vel, -10, 10)
            self.pos = np.add(self.vel, self.pos)
            self.counter += 1
            # check if out of bounds
            if self.pos[0] <= 0 or self.pos[0] >= WIDTH-self.size:
                self.dead = True
            if self.pos[1] <= 0 or self.pos[1] >= HEIGHT-self.size:
                self.dead = True
            # check collision
            for obstacle in obstacles.list:
                if obstacle.collidepoint(self.pos):
                    self.dead = True
            # check if reached target
            if math.sqrt((target[0] - self.pos[0])**2 + (target[1] - self.pos[1])**2) <= target_rad:
                self.win = True
                self.dead = True
            if self.dead:
                self.calc_fit()

    def calc_fit(self):
        if self.win:
            self.fitness = (1/self.counter) * num_of_moves * 100
        else:
            self.fitness = (1 / (math.sqrt(((target[0] - self.pos[0])**2) + (
                (target[1] - self.pos[1])**2)))) * 1000 + self.counter/num_of_moves
        return self.fitness


class Population:
    def __init__(self, pop_size=population_size, members=None):
        if members is None:
            members = []
        self.members = members
        self.size = pop_size

    def initialize(self):
        if not self.members:
            for i in range(self.size):
                self.members.append(Individual(int(WIDTH/2), HEIGHT-20))
            for i in self.members:
                i.generate_DNA()
        else:
            new_members = []
            for i in self.members:
                new_members.append(Individual(
                    int(WIDTH/2), HEIGHT-20, DNA=i.DNA))
            self.members = new_members
            self.size = len(self.members)

    def update(self):
        for i in self.members:
            i.update()

    def draw(self):
        for i in self.members:
            i.draw()

    def all_dead(self):
        dead = []
        for i in self.members:
            dead.append(i.dead)
        if False in dead:
            return False
        else:
            return True

    def select_cross(self):
        # select
        cross_members = []
        fitnesses = [i.fitness for i in self.members]
        fitnesses_members = [(i.fitness, i) for i in self.members]
        fitnesses.sort()
        fitnesses_members.sort(key=operator.itemgetter(0))
        self.members = [fitnesses_members[i][1] for i in range(len(fitnesses))]
        fitness_sum = float(sum(fitnesses))
        rel_fitness = [f/fitness_sum for f in fitnesses]
        probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
        for i in range(10):
            cross_members.append(self.members[-1])
        while len(cross_members) < population_size:
            r = random.random()
            for i, member in enumerate(self.members):
                if r <= probs[i]:
                    cross_members.append(member)
                    break
        global max_fit
        max_fit = self.members[-1].DNA
        global max_fitness
        max_fitness = self.members[-1].fitness

        # cross
        new_members = [self.members[-1]]
        for i in range(population_size):
            rand1 = cross_members[random.randint(0, len(cross_members) - 1)]
            rand2 = cross_members[random.randint(0, len(cross_members) - 1)]
            if rand1 == rand2:
                new_members.append(rand1)
            else:
                DNA1, DNA2 = rand1.DNA, rand2.DNA
                cross_index = random.randint(1, len(DNA1))
                childDNA = DNA1[:cross_index] + DNA2[cross_index:]
                child = Individual(0, 0, DNA=childDNA)
                new_members.append(child)
        return new_members


def mutate(pop):
    for i in pop:
        for j in i.DNA:
            if random.random() <= mutation_rate:
                i.DNA[random.randint(
                    0, num_of_moves-1)] = np.array([random.randint(-3, 3), random.randint(-3, 3)])


pop = Population()
pop.initialize()

target = [int(WIDTH / 2), 20]
target_rad = 10

generation = 1
max_fit_dot = None
max_fitness = 0

toggle_mode = False
if __name__ == "__main__":
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if toggle_mode:
                    if event.key == pygame.K_t:
                        toggle_mode = False
                elif not toggle_mode:
                    if event.key == pygame.K_t:
                        toggle_mode = True

        display.fill(bgcolor)

        pygame.draw.circle(display, GREEN, target, target_rad)
        obstacles.draw()

        if pop.all_dead():
            new_pop = pop.select_cross()
            max_fit_real = [i for i in max_fit]
            max_fit_dot = Individual(int(WIDTH/2), HEIGHT-20, DNA=max_fit_real)
            max_fit_dot.color = RED
            max_fit_dot.size = 5
            mutate(new_pop)
            pop = Population(members=new_pop)
            pop.initialize()
            generation += 1

        pop.update()
        if max_fit_dot:
            max_fit_dot.update()
        if toggle_mode:
            if max_fit_dot:
                max_fit_dot.draw()
        else:
            pop.draw()
            if max_fit_dot:
                max_fit_dot.draw()

        display.blit(arial.render("Generation: " + str(generation),
                     1, (255, 255, 255)), (15, 10))
        display.blit(arial.render(
            "Max Fitness: " + str(round(max_fitness, 2)), 1, (255, 255, 255)), (WIDTH-230, 10))
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

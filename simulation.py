import numpy as np
import random
import pygame
import torch
import torch.nn as nn
from collections import defaultdict

# Constants
FIELD_SIZE = 160
NUM_CREATURES = 80
MUTATION_RATE = 0.25
INITIAL_ENERGY = 200
ENERGY_GAIN_FROM_FOOD = 150
NUM_FOOD_ITEMS = 300
TICKS_BEFORE_SPLIT = 225
ENERGY_TO_SPLIT = 20
VISION_RADIUS = 50
WINDOW_SIZE = 800
CELL_SIZE = WINDOW_SIZE // FIELD_SIZE
MAX_SPEED = 0.5
MAX_ANGULAR_VELOCITY = np.pi * 2
FPS = 30
FOOD_EATING_THRESHOLD = 0.5
GRID_SIZE = 10
FOOD_ADDITION_RATE = 0.1

class CreatureBrain(nn.Module):
    def __init__(self):
        super(CreatureBrain, self).__init__()
        self.hidden1 = nn.Linear(4, 10)
        self.output = nn.Linear(10, 3)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.hidden1(x))
        return self.output(x)

class Creature:
    def __init__(self):
        self.brain = CreatureBrain()
        self.position = np.array([random.uniform(0, FIELD_SIZE - 1), random.uniform(0, FIELD_SIZE - 1)])
        self.energy = INITIAL_ENERGY
        self.angle = random.uniform(0, 2 * np.pi)
        self.color = self.get_color_from_weights()
        self.alive = True
        self.ticks_lived = 0
        self.last_split_ticks = 0

    def get_color_from_weights(self):
        weights = np.concatenate([p.detach().numpy().flatten() for p in self.brain.parameters()])
        
        min_val, max_val = np.min(weights), np.max(weights)
        if max_val - min_val == 0:
            normalized_weights = np.zeros_like(weights)
        else:
            normalized_weights = (weights - min_val) / (max_val - min_val)
        
        color = (normalized_weights[:3] * 255).astype(int)
        return tuple(np.clip(color, 0, 255))

    def get_inputs(self, nearby_food):
        visible_food = self.get_visible_food(nearby_food)
        if visible_food:
            closest_food = min(visible_food, key=lambda f: np.linalg.norm(self.position - np.array(f)))
            distance_to_food = np.linalg.norm(np.array(closest_food) - self.position) / FIELD_SIZE
            angle_to_food = self.calculate_angle_to_food(closest_food)
        else:
            distance_to_food = 1
            angle_to_food = 0

        return torch.tensor([distance_to_food, angle_to_food, self.energy / INITIAL_ENERGY, 1], dtype=torch.float32).view(1, -1)

    def calculate_angle_to_food(self, food_position):
        food_vector = np.array(food_position) - self.position
        angle_to_food = np.arctan2(food_vector[1], food_vector[0])
        relative_angle = (angle_to_food - self.angle + np.pi) % (2 * np.pi) - np.pi
        return relative_angle / np.pi

    def get_visible_food(self, nearby_food):
        visible_food = []
        for food in nearby_food:
            food_vector = np.array(food) - self.position
            angle_to_food = np.arctan2(food_vector[1], food_vector[0])
            relative_angle = (angle_to_food - self.angle + np.pi) % (2 * np.pi) - np.pi
            if np.abs(relative_angle) <= np.pi / 4:
                visible_food.append(food)
        return visible_food

    def decide_movement(self, nearby_food):
        input_vector = self.get_inputs(nearby_food)
        output = self.brain(input_vector)
        
        speed = torch.sigmoid(output[0, 0]).item() * MAX_SPEED
        angular_velocity = torch.tanh(output[0, 1]).item() * MAX_ANGULAR_VELOCITY
        
        reproduction_threshold = torch.sigmoid(output[0, 2]).item()
        should_reproduce = reproduction_threshold > 0.5
        
        return speed, angular_velocity, should_reproduce

    def move(self, creatures, grid, food_positions):
        if self.energy <= 0:
            self.alive = False
            return False

        nearby_food = get_nearby_food(grid, self.position)
        speed, angular_velocity, should_reproduce = self.decide_movement(nearby_food)
        self.angle += angular_velocity
        new_position = np.clip(self.position + speed * np.array([np.cos(self.angle), np.sin(self.angle)]), 0, FIELD_SIZE - 1)

        energy_depletion = 0.5 + np.exp(speed)
        self.energy -= energy_depletion

        for food in list(nearby_food):
            food_array = np.array(food)
            distance_to_food = np.linalg.norm(new_position - food_array)
            if distance_to_food <= FOOD_EATING_THRESHOLD:
                self.energy += ENERGY_GAIN_FROM_FOOD
                remove_from_grid(grid, food)
                food_positions.discard(food)

        if should_reproduce and self.energy > ENERGY_TO_SPLIT and self.can_split():
            self.energy -= ENERGY_TO_SPLIT
            creatures.append(split(self))

        if not any(np.array_equal(new_position, c.position) for c in creatures if c != self):
            self.position = new_position

        self.ticks_lived += 1
        return self.energy > 0

    def can_split(self):
        return (self.ticks_lived - self.last_split_ticks) >= TICKS_BEFORE_SPLIT

def split(creature):
    child = Creature()
    child.brain.load_state_dict(creature.brain.state_dict())
    mutate(child)
    
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    random.shuffle(neighbors)
    for dx, dy in neighbors:
        new_position = creature.position + np.array([dx, dy])
        if 0 <= new_position[0] < FIELD_SIZE and 0 <= new_position[1] < FIELD_SIZE:
            child.position = new_position
            break
    else:
        child.position = creature.position

    creature.last_split_ticks = creature.ticks_lived
    return child

def mutate(creature):
    for p in creature.brain.parameters():
        mutation_mask = torch.rand_like(p) < MUTATION_RATE
        p.data += mutation_mask.float() * torch.normal(0, 0.1, p.size())
    creature.color = creature.get_color_from_weights()

def add_to_grid(grid, position):
    grid_key = (int(position[0] // GRID_SIZE), int(position[1] // GRID_SIZE))
    grid[grid_key].add(tuple(position))

def remove_from_grid(grid, position):
    grid_key = (int(position[0] // GRID_SIZE), int(position[1] // GRID_SIZE))
    if grid_key in grid:
        grid[grid_key].discard(tuple(position))

def get_nearby_food(grid, position):
    grid_key = (int(position[0] // GRID_SIZE), int(position[1] // GRID_SIZE))
    nearby_food = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
            if neighbor_key in grid:
                nearby_food.update(grid[neighbor_key])
    return nearby_food

def place_food(grid):
    food_positions = set()
    while len(food_positions) < NUM_FOOD_ITEMS:
        food_pos = (random.uniform(0, FIELD_SIZE - 1), random.uniform(0, FIELD_SIZE - 1))
        add_to_grid(grid, food_pos)
        food_positions.add(food_pos)
    return food_positions

def add_random_food(grid, food_positions):
    if random.random() < FOOD_ADDITION_RATE:
        food_pos = (random.uniform(0, FIELD_SIZE - 1), random.uniform(0, FIELD_SIZE - 1))
        add_to_grid(grid, food_pos)
        food_positions.add(food_pos)

def draw(simulation_surface, creatures, food_positions):
    simulation_surface.fill((255, 255, 255))
    for food in food_positions:
        pygame.draw.rect(simulation_surface, (255, 0, 0), (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for creature in creatures:
        if creature.alive:
            pygame.draw.rect(simulation_surface, creature.color, (creature.position[0] * CELL_SIZE, creature.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()

def simulate():
    creatures = [Creature() for _ in range(NUM_CREATURES)]
    grid = defaultdict(set)
    food_positions = place_food(grid)

    pygame.init()
    simulation_surface = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if all(not creature.alive for creature in creatures):
            creatures = [Creature() for _ in range(NUM_CREATURES)]

        creatures = [creature for creature in creatures if creature.alive]

        for creature in creatures:
            if creature.alive:
                creature.move(creatures, grid, food_positions)
                if creature.can_split():
                    creatures.append(split(creature))

        draw(simulation_surface, creatures, food_positions)
        clock.tick(FPS)
        add_random_food(grid, food_positions)

    pygame.quit()

if __name__ == "__main__":
    simulate()
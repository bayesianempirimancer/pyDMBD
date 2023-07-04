import random
import math
import torch
import matplotlib.pyplot as plt


class Forager:
    def __init__(self):
        # Set the number of foods and their properties
        self.num_foods = 20
        self.food_range = 100
        self.forager_speed = 1
        self.vision_range = 25
        self.max_food_items = 3
        self.d_max = 75  # Maximum distance from home
        self.num_steps = 20000

    def simulate(self):
        num_foods = self.num_foods
        food_range = self.food_range
        forager_speed = self.forager_speed
        vision_range = self.vision_range
        max_food_items = self.max_food_items
        d_max = self.d_max
        num_steps = self.num_steps

        # Create a list to store the foods and their initial locations
        foods = [(random.uniform(-food_range, food_range), random.uniform(-food_range, food_range))
                 for _ in range(num_foods)]

        # Set the initial location of the forager and its memory
        forager_location = (0, 0)
        forager_memory = []
        food_collected = 0

        # Store the positions of forager, food items, consumed food, and food memory at each time point
        forager_positions = [forager_location]
        food_positions = [foods[:]]
        consumed_food_positions = [[]]
        food_memory = []

        # Simulate the foraging process
        for step in range(num_steps):
            if food_collected == max_food_items:
                forager_location = (0, 0)
                food_collected = 0
                forager_memory.clear()
                food_memory.clear()

            if food_collected < max_food_items:
                # Check if the forager sees any new food items
                new_food_items = [food for food in foods if food not in forager_memory and
                                  math.sqrt((forager_location[0] - food[0]) ** 2 +
                                            (forager_location[1] - food[1]) ** 2) <= vision_range]

                if new_food_items:
                    # Add the new food items to the forager's memory and mark them as present in memory
                    forager_memory.extend(new_food_items)
                    food_memory.extend([1] * len(new_food_items))

                if forager_memory:
                    # Find the nearest food item
                    nearest_food = min(forager_memory, key=lambda f: math.sqrt((forager_location[0] - f[0]) ** 2 +
                                                                                (forager_location[1] - f[1]) ** 2))

                    # Calculate the angle between the forager and the nearest food item
                    angle = math.atan2(nearest_food[1] - forager_location[1], nearest_food[0] - forager_location[0])

                    # Move towards the nearest food item with the maximum forager speed
                    forager_location = (
                        forager_location[0] + forager_speed * math.cos(angle),
                        forager_location[1] + forager_speed * math.sin(angle)
                    )

                    # Check if the forager has reached the nearest food item
                    if math.sqrt((forager_location[0] - nearest_food[0]) ** 2 +
                                (forager_location[1] - nearest_food[1]) ** 2) <= forager_speed:
                        # Move the food item to a new random location
                        foods.remove(nearest_food)
                        new_location = (random.uniform(-food_range, food_range),
                                        random.uniform(-food_range, food_range))
                        foods.append(new_location)

                        # Remove the food item from the forager's memory and mark it as absent in memory
                        index = forager_memory.index(nearest_food)
                        forager_memory.pop(index)
                        food_memory.pop(index)
                        food_collected += 1
                        consumed_food_positions[-1].append(nearest_food)
                else:
                    # Randomly move away from home until reaching d_max distance
                    if math.sqrt(forager_location[0] ** 2 + forager_location[1] ** 2) <= d_max:
                        angle = random.uniform(0, 2 * math.pi) / 4.0
                        forager_location = (
                            forager_location[0] + forager_speed * math.cos(angle),
                            forager_location[1] + forager_speed * math.sin(angle)
                        )
                    else:
                        # Move clockwise around home until seeing a new food item
                        angle = math.atan2(forager_location[1], forager_location[0]) + math.pi / 2
                        forager_location = (
                            forager_location[0] + forager_speed * math.cos(angle),
                            forager_location[1] + forager_speed * math.sin(angle)
                        )

            # Store the positions of forager, food items, consumed food, and food memory at each time point
            forager_positions.append(forager_location)
            food_positions.append(foods[:])
            consumed_food_positions.append([])
            food_memory.append([0] * len(foods))

        return forager_positions, food_positions, consumed_food_positions, food_memory

    def plot(self, forager_positions, food_positions, consumed_food_positions):
        # Extract x and y coordinates for plotting
        forager_x, forager_y = zip(*forager_positions)
        food_x, food_y = zip(*food_positions)
        consumed_food_x, consumed_food_y = zip(*consumed_food_positions)

        # Plot the trajectory of the forager, food locations, and consumed food locations
        plt.figure(figsize=(8, 6))
        plt.plot(forager_x, forager_y, label="Forager Trajectory")
        plt.scatter(food_x, food_y, marker='o', color='red', label="Food Locations")
        plt.scatter(consumed_food_x, consumed_food_y, marker='x', color='green', label="Consumed Food Locations")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Forager Trajectory and Food Locations")
        plt.legend()
        plt.grid(True)
        plt.show()


sim = Forager()
forager_positions, food_positions, consumed_food_positions, food_memory = sim.simulate()
sim.plot(forager_positions, food_positions, consumed_food_positions)

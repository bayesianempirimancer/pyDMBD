import random
import math
import torch
import matplotlib.pyplot as plt


class Forager():

    def __init__(self):
        # Set the number of foods and their properties
        self.num_foods = 10
        self.food_range = 100
        self.forager_speed = 1
        self.vision_range = 20
        self.max_food_items = 3
        self.d_max = 75  # Maximum distance from home
        self.num_steps = 2000
        self.noise = 0.5

    def simulate(self):
        num_foods = self.num_foods
        food_range = self.food_range
        forager_speed = self.forager_speed
        vision_range = self.vision_range
        max_food_items = self.max_food_items
        d_max = self.d_max
        num_steps = self.num_steps
        noise = self.noise
    # Create a list to store the foods and their initial locations
        foods = [(random.uniform(-food_range, food_range), random.uniform(-food_range, food_range))
                for _ in range(num_foods)]

        # Set the initial location of the forager and its memory
        forager_location = (0, 0)
        forager_memory = []
        food_collected = 0
        food_in_memory = [0] * self.num_foods

        # Store the positions of forager, food items, and consumed food at each time point
        forager_positions = [forager_location]
        food_positions = [foods[:]]
        food_memory = [food_in_memory[:]]

        rand_direction = 2*math.pi*random.uniform(0,1)

        # Simulate the foraging process
        for step in range(num_steps):

            new_food_items = [food for food in foods if food not in forager_memory and
                            math.sqrt((forager_location[0] - food[0]) ** 2 +
                                        (forager_location[1] - food[1]) ** 2) <= vision_range]

            if new_food_items:
                # Add the new food items to forager's memory
                forager_memory.extend(new_food_items)
                for food in new_food_items:
                    food_in_memory[foods.index(food)] = 1

            if food_collected == max_food_items:
                # Move back towards (0, 0) at normal speed
                angle = math.atan2(-forager_location[1], -forager_location[0])
                forager_location = (
                    forager_location[0] + forager_speed * math.cos(angle) + random.normalvariate(0, noise),
                    forager_location[1] + forager_speed * math.sin(angle) + random.normalvariate(0, noise)
                )

                # Check if the forager has reached the origin
                if math.sqrt(forager_location[0] ** 2 + forager_location[1] ** 2) <= forager_speed:
                    # Reset food_collected and forager_memory
                    food_collected = 0 
                    rand_direction = 2*math.pi*random.uniform(0,1)

                # Check if the forager sees any new food items

            if food_collected < max_food_items:
                if forager_memory:
                    # Find the nearest food item
                    nearest_food = min(forager_memory, key=lambda f: math.sqrt((forager_location[0] - f[0]) ** 2 +
                                                                            (forager_location[1] - f[1]) ** 2))

                    # Calculate the angle between forager and nearest food item
                    angle = math.atan2(nearest_food[1] - forager_location[1], nearest_food[0] - forager_location[0])

                    # Move towards the nearest food item with maximum forager speed
                    forager_location = (
                        forager_location[0] + forager_speed * math.cos(angle) + random.normalvariate(0, noise),
                        forager_location[1] + forager_speed * math.sin(angle) + random.normalvariate(0, noise)
                    )

                    # Check if the forager has reached the nearest food item
                    if math.sqrt((forager_location[0] - nearest_food[0]) ** 2 +
                                (forager_location[1] - nearest_food[1]) ** 2) <= forager_speed:
                        # Move the food item to a new random location
                        food_in_memory[foods.index(nearest_food)] = 0
                        foods[foods.index(nearest_food)] = (random.uniform(-food_range, food_range), random.uniform(-food_range, food_range))

                        # Remove the food item from forager's memory
                        forager_memory.remove(nearest_food)
                        food_collected += 1

                else:
                    # Randomly move away from home until reaching d_max distance
                    if math.sqrt(forager_location[0] ** 2 + forager_location[1] ** 2) <= d_max:
                        angle = rand_direction
                        forager_location = (
                            forager_location[0] + forager_speed * math.cos(angle) + random.normalvariate(0, noise),
                            forager_location[1] + forager_speed * math.sin(angle) + random.normalvariate(0, noise)
                        )
                    else:
                        # Move clockwise around home until seeing a new food item
                        angle = math.atan2(forager_location[1], forager_location[0]) + math.pi / 2
                        forager_location = (
                            forager_location[0] + forager_speed * math.cos(angle) + random.normalvariate(0, noise),
                            forager_location[1] + forager_speed * math.sin(angle) + random.normalvariate(0, noise)
                        )

            # Store the positions of forager, food items, and consumed food at each time point
            forager_positions.append(forager_location)
            food_positions.append(foods[:])
            food_memory.append(food_in_memory[:])

        return torch.tensor(forager_positions), torch.tensor(food_positions), torch.tensor(food_memory)

    def plot(self, forager_positions, food_positions):

        # Plot the trajectory of the forager, food locations, and consumed food locations
        plt.figure(figsize=(8, 6))
        plt.plot(forager_positions[:,0], forager_positions[:,1], label="Forager Trajectory")
        plt.scatter(food_positions[:,:,0], food_positions[:,:,1], marker='x', color='green', label="Consumed Food Locations")
        plt.scatter(food_positions[-1,:,0], food_positions[-1,:,1], marker='o', s=80, color='red', label="Remaining Food Locations")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Forager Trajectory and Food Locations")
        plt.grid(True)
        plt.show()

    def simulate_batches(self, batch_num):
        forager_positions = torch.zeros(sim.num_steps + 1,batch_num,2)
        food_positions = torch.zeros(sim.num_steps + 1,batch_num,sim.num_foods,2)
        food_memory = torch.zeros(sim.num_steps + 1,batch_num,sim.num_foods)

        for i in range(0,batch_num):
            forager_positions[:,i,:], food_positions[:,i,:,:], food_memory[:,i,:] = sim.simulate()

        data = torch.cat((forager_positions.unsqueeze(-2),food_positions),-2)

        return data, food_memory


sim = Forager()
forager_positions, food_positions, food_memory = sim.simulate()
sim.plot(forager_positions,food_positions)

# batch_num = 100
# data = sim.simulate_batches(batch_num)

# v_data = data.diff(n=1,dim=0)
# data = data[1:]

# v_data = torch.cat((data,v_data),-1)
# v_data = v_data + torch.randn(v_data.shape)*0.1
# v_data = v_data/v_data.std((0,1),True)
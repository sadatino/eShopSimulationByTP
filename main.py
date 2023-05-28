
import numpy as np
import random
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule




gotten_data = []

penalty_factor = 82
similarity_threshold = 0.72
neighbor_success_prob = 0.6

MINIMUM_SELLERS_TO_VIEW = 5
NUMBER_STEPS_TO_WRITE = 5
NUMBER_STEPS_TO_WRITE_OPTIMIZATION = 3

class Product:
    def __init__(self, search_by_query_name, seller_feedback, sales_history, time_spent_on_product_page,seller_response,
                 delivery_speed, search_relevance, click_rate, conversion, views, shopping_cart):
        self.seller_feedback = seller_feedback
        self.search_relevance = search_relevance
        self.seller_response = seller_response
        self.delivery_speed = delivery_speed
        self.sales_history = sales_history
        self.time_spent_on_product_page = time_spent_on_product_page
        self.click_rate = click_rate
        self.conversion = conversion
        self.views = views
        self.shopping_cart = shopping_cart
        self.search_by_query_name = search_by_query_name


class Buyer(Agent):
    def __init__(self, unique_id, model, attribute_values):
        super().__init__(unique_id, model)
        self.attribute_values = attribute_values
        self.successful_seller = None
        self.number_of_sellers_to_view = random.randint(self.model.min_products_to_view, total_sellers)

    def review_seller(self, seller):
        review_chance = random.random()
        if review_chance > 0.8:
            for product in seller.products:
                product.search_by_query_name += 0.00000000000000001
                product.seller_feedback += 0.00000000000000001
                product.sales_history += 0.00000000000000001
                product.time_spent_on_product_page += 0.00000000000000001
        elif review_chance > 0.6:
            attributes_to_update = random.sample(range(4), 2)
            for product in seller.products:
                for attr_index in attributes_to_update:
                    if attr_index == 0:
                        product.search_by_query_name += 0.00000000000000001
                    elif attr_index == 1:
                        product.seller_feedback += 0.00000000000000001
                    elif attr_index == 2:
                        product.sales_history += 0.00000000000000001
                    elif attr_index == 3:
                        product.time_spent_on_product_page += 0.00000000000000001

        elif review_chance < 0.2:
            seller.products[0].seller_feedback -= 0.00000000000000001
            seller.products[0].sales_history -= 0.00000000000000001

        elif review_chance < 0.4:
            attributes_to_update = random.sample(range(4), 3)
            for product in seller.products:
                for attr_index in attributes_to_update:
                    if attr_index == 0:
                        product.search_by_query_name -= 0.00000000000000001
                    elif attr_index == 1:
                        product.seller_feedback -= 0.00000000000000001
                    elif attr_index == 2:
                        product.sales_history -= 0.00000000000000001
                    elif attr_index == 3:
                        product.time_spent_on_product_page -= 0.00000000000000001

    def step(self):

        # Get the neighbors of the buyer
        buyer_pos = self.pos
        neighbors = self.model.grid.get_neighbors(buyer_pos, moore=True, include_center=False, radius=1)

        sorted_sellers = self.model.sorted_sellers_per_step

        sorted_sellers_final = sorted_sellers[:self.number_of_sellers_to_view]


        # Try to buy from sellers
        for seller in sorted_sellers_final:
            sold = False
            self.model.interactions_per_step += 1

            # Check if any neighbor had a successful interaction with the current seller

            neighbor_success = any(neighbor.successful_seller == seller.unique_id for neighbor in neighbors if
                                   isinstance(neighbor, Buyer)) and random.random() < neighbor_success_prob

            # Calculate the cosine similarity between the buyer's criteria weights and product weights
            similarities = []
            for product in seller.products:
                product_values = [product.seller_feedback, product.search_relevance, product.seller_response,
                                  product.delivery_speed, product.sales_history, product.time_spent_on_product_page,
                                  product.click_rate, product.conversion,product.views,product.shopping_cart,
                                  product.search_by_query_name]
                #similarity = cosine_similarity(self.attribute_values, product_values)
                #
                # ###### euclidean distance
                # distance = euclidean_distance(self.attribute_values, product_values)
                # max_distance = math.sqrt(len(product_values))
                # similarity = 1 - (distance / max_distance)

                similarity = minkowski_similarity(self.attribute_values[:5], product_values[:5])

                similarities.append(similarity)

            # Determine the most similar product
            # sorted_numbers_with_indexes = get_max_three(similarities)
            #
            # random_number = random.randint(1, 3)
            #
            # chosen_product_index = sorted_numbers_with_indexes[random_number-1][0]
            # max_similarity = sorted_numbers_with_indexes[random_number-1][1]
            # pasirenkamas max similarity produktas
            max_similarity = max(similarities)
            chosen_product_index = similarities.index(max_similarity)

            # If the similarity is above the threshold (e.g., 0.8) or a neighbor had a successful interaction, buy the product
            if max_similarity >= self.model.success_threshold:

                seller.sales += 1
                seller.products[chosen_product_index].sales_history += 0.00000000000000001
                self.model.successful_transactions += 1
                self.review_seller(seller)  # Call the review_seller method after a successful transaction
                self.successful_seller = seller.unique_id  # Update the successful_seller attribute
                sold = True
                break
            elif neighbor_success and max_similarity <= similarity_threshold:
                seller.sales += 1
                seller.products[chosen_product_index].sales_history += 0.00000000000000001
                self.model.successful_transactions += 1
                self.review_seller(seller)  # Call the review_seller method after a successful transaction

                #print("KAIMYNAS")

                # Track the sale
                sale_key = f"Seller {seller.unique_id}, Product {chosen_product_index}"
                if sale_key not in self.model.tracked_sales:
                    self.model.tracked_sales[sale_key] = 0
                self.model.tracked_sales[sale_key] += 1
                sold = True
                break

        if sold == False:
            self.model.missed_transactions += 1



class Seller(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.products = self.create_products()
        self.sales = 0
        self.score_for_sorting = 0

    def create_products(self):
        # Create products with random weights between 0 and 1
        products = []
        for _ in range(1):  # Number of products for each seller
            values = [random.random() for _ in range(11)]
            product = Product(*values)
            products.append(product)
        return products

    def step(self):
        pass

class ShopOwner(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        pass

class OnlineShopModel(Model):
    def __init__(self, n_buyers, n_sellers, n_success_threshold, n_minimum_amount_products_view, seller_feedback_lower,
                 seller_feedback_upper, search_relevance_lower, search_relevance_upper, seller_response_lower,
                 seller_response_upper, delivery_speed_lower, delivery_speed_upper, sales_history_lower,
                 sales_history_upper, width, height):
        self.optimizing = False
        self.optimizing_value = 0
        self.tracked_sales = {}
        self.num_buyers = n_buyers
        self.num_sellers = n_sellers
        self.min_products_to_view = n_minimum_amount_products_view
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        self.running = True
        self.missed_transactions = 0
        self.successful_transactions = 0
        self.interactions_per_step = 0
        self.succ_transactions = 0

        self.success_threshold = n_success_threshold

        self.seller_feedback_lower = seller_feedback_lower
        self.seller_feedback_upper = seller_feedback_upper
        self.search_relevance_lower = search_relevance_lower
        self.search_relevance_upper = search_relevance_upper
        self.seller_response_lower = seller_response_lower
        self.seller_response_upper = seller_response_upper
        self.delivery_speed_lower = delivery_speed_lower
        self.delivery_speed_upper = delivery_speed_upper
        self.sales_history_lower = sales_history_lower
        self.sales_history_upper = sales_history_upper

        self.criteria_weights = [0.3,0.01,0.01,0.01,0.05,0.01,0.01,0.01,0.1,0.15,0.34]

        self.sorted_sellers_per_step = []


        self.datacollectorSales = {}
        self.datacollector = DataCollector(
            model_reporters={
                "Successful_Sales": lambda m: m.succ_transactions,
                "Seller_Sales": lambda m: {f"Seller {seller.unique_id}": seller.sales for seller in m.schedule.agents if
                                           isinstance(seller, Seller)}}
        )

        # Create buyers
        for i in range(self.num_buyers):
            buyer = Buyer(i, self, self.generate_random_values(11))  # AttributeCount
            self.schedule.add(buyer)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(buyer, (x, y))

        # Create sellers
        for i in range(self.num_sellers):
            seller = Seller(self.num_buyers + i, self)
            self.schedule.add(seller)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(seller, (x, y))

        # Create shop owner
        shop_owner = ShopOwner(self.num_buyers + self.num_sellers, self)
        self.schedule.add(shop_owner)
        x = random.randrange(self.grid.width)
        y = random.randrange(self.grid.height)
        self.grid.place_agent(shop_owner, (x, y))


    def optimize_weights(self):
        sellers = [agent for agent in self.schedule.agents if isinstance(agent, Seller)]
        self.successful_transactions = 0

        # Fitness function to evaluate the sellers
        def fitness(weights):
            # Temporarily update the model's criteria_weights with the new weights
            original_weights = self.criteria_weights
            self.criteria_weights = weights

            # Run a few steps of the model to simulate transactions with the new weights
            self.optimizing = True

            for _ in range(NUMBER_STEPS_TO_WRITE_OPTIMIZATION):
                self.step()
                self.optimizing_value = 1

            self.optimizing = False
            self.optimizing_value = 0

            # Calculate the total sales with the new weights
            total_sales = self.successful_transactions

            zero_sales_sellers = sum(1 for seller in sellers if seller.sales == 0)

            # Reset the model's criteria_weights and the successful_transactions counter
            self.criteria_weights = original_weights
            self.successful_transactions = 0

            return max(0, total_sales - (self.num_buyers / self.num_sellers) * zero_sales_sellers)


        # Genetic Algorithm implementation
        def genetic_algorithm(fitness_function, weights, population_size=30, generations=25, crossover_rate=0.8,
                              mutation_rate=0.1, convergence_generations=5):
            def create_individual():

                # Generate 11 random values
                values = np.random.random(11)

                # Calculate the total
                total = np.sum(values)

                # Divide each value by the total
                values /= total

                return values

            def mutate(individual, epsilon=1e-6):
                i = random.randint(0, len(individual) - 1)
                delta = random.uniform(-individual[i] + epsilon, individual[i] - epsilon)
                individual[i] += delta

                j = random.randint(0, len(individual) - 2)
                if j >= i:
                    j += 1

                delta_j = min(individual[j], -delta)  # Ensure the new value for individual[j] is non-negative
                individual[j] -= delta_j

                # Normalize the weights
                total = sum(individual)
                individual = [w / total for w in individual]


                return individual

            def crossover(parent1, parent2):
                crossover_point = np.random.randint(1, len(parent1))
                if isinstance(parent1, np.ndarray):
                    parent1 = parent1.tolist()

                if isinstance(parent2, np.ndarray):
                    parent2 = parent2.tolist()

                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

                # Normalize the weights
                total1 = sum(child1)
                child1 = [w / total1 for w in child1]

                total2 = sum(child2)
                child2 = [w / total2 for w in child2]


                return child1, child2

            population = [create_individual() for _ in range(population_size)]

            best_fitness_so_far = -1
            num_generations_no_improvement = 0

            for gen in range(generations):
                fitness_values = [fitness_function(individual) for individual in population]
                new_population = []

                current_best_fitness = max(fitness_values)
                if current_best_fitness > best_fitness_so_far:
                    best_fitness_so_far = current_best_fitness
                    num_generations_no_improvement = 0
                else:
                    num_generations_no_improvement += 1

                if num_generations_no_improvement >= convergence_generations:
                    break

                while len(new_population) < population_size:
                    selected_indices = np.random.choice(range(len(population)), size=2,
                                                        p=np.array(fitness_values) / (sum(fitness_values) + 1e-10))

                    parent1, parent2 = population[selected_indices[0]], population[selected_indices[1]]

                    if random.random() < crossover_rate:
                        child1, child2 = crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1, parent2

                    if random.random() < mutation_rate:
                        child1 = mutate(child1)

                    if random.random() < mutation_rate:
                        child2 = mutate(child2)

                    new_population.extend([child1, child2])
                print("generation")
                population = new_population

            fitness_values = [fitness_function(individual) for individual in population]
            best_individual = population[np.argmax(fitness_values)]
            return best_individual

        # Update the weights with the Genetic Algorithm
        new_weights = genetic_algorithm(fitness, self.criteria_weights)
        self.criteria_weights = new_weights

    def step(self):
        if self.schedule.steps == 0 and not self.optimizing:
            self.sorted_sellers_per_step = self.calculate_score_return_sorted_sellers()

        elif self.schedule.steps % NUMBER_STEPS_TO_WRITE_OPTIMIZATION == 0 and self.optimizing:
            if self.optimizing_value == 0:
                self.sorted_sellers_per_step = self.calculate_score_return_sorted_sellers()
            if self.schedule.steps != 0:
                self.schedule.steps = 0

            for agent in self.schedule.agents[self.num_buyers - 1:]:
                if isinstance(agent, Seller):  # Reset the sales count for each seller
                    agent.sales = 0

        self.interactions_per_step = 0
        self.schedule.step()
        self.datacollector.collect(self)

        if self.schedule.steps == 50 and not self.optimizing:  # Stop the model after 2000 steps (backup condition)
            self.running = False
            min_sellers = min(gotten_data, key=lambda x: x[0])[0]
            max_successful_transactions = max(
                filter(lambda x: x[0] == min_sellers, gotten_data),
                key=lambda x: x[1]
            )
            index_max_success = max(range(len(gotten_data)), key=lambda i: gotten_data[i][1])

            weights_minimum_0_sales = max_successful_transactions[2]
            weights_max_success = gotten_data[index_max_success][2]

            print("Minimum amount of sellers with 0 sales. Weights are: ",
                  weights_minimum_0_sales[0], weights_minimum_0_sales[1], weights_minimum_0_sales[2], weights_minimum_0_sales[3],
                  weights_minimum_0_sales[4], weights_minimum_0_sales[5], weights_minimum_0_sales[6], weights_minimum_0_sales[7],
                  weights_minimum_0_sales[8], weights_minimum_0_sales[9], weights_minimum_0_sales[10]
                  )
            result_string = "Minimum amount of sellers with 0 sales. Weights are: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                weights_minimum_0_sales[0], weights_minimum_0_sales[1], weights_minimum_0_sales[2],
                weights_minimum_0_sales[3], weights_minimum_0_sales[4], weights_minimum_0_sales[5],
                weights_minimum_0_sales[6], weights_minimum_0_sales[7], weights_minimum_0_sales[8],
                weights_minimum_0_sales[9], weights_minimum_0_sales[10]
            )

            write_string_to_file('results.txt', result_string)
            print("Maximum amount of successful transactions. Weights are: ",
                  weights_max_success[0], weights_max_success[1], weights_max_success[2], weights_max_success[3],
                  weights_max_success[4], weights_max_success[5], weights_max_success[6], weights_max_success[7],
                  weights_max_success[8], weights_max_success[9], weights_max_success[10]
                  )
            result_string2 = "Maximum amount of successful transactions. Weights are: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                weights_max_success[0], weights_max_success[1], weights_max_success[2], weights_max_success[3],
                weights_max_success[4], weights_max_success[5], weights_max_success[6], weights_max_success[7],
                weights_max_success[8], weights_max_success[9], weights_max_success[10]
                )
            write_string_to_file('results.txt', result_string2)



        elif self.schedule.steps % NUMBER_STEPS_TO_WRITE == 0 and self.schedule.steps != 0 and not self.optimizing:
            self.write_best_seller_to_file()

            saveStepsAvoidOptimization = self.schedule.steps
            self.schedule.steps = 0
            self.optimize_weights()
            self.schedule.steps = saveStepsAvoidOptimization

            for agent in self.schedule.agents[self.num_buyers-1:]:
                if isinstance(agent, Seller):  # Reset the sales count for each seller
                    agent.sales = 0



            # after each step, make the transactions 0
            self.missed_transactions = 0
            self.sorted_sellers_per_step = self.calculate_score_return_sorted_sellers()


    def calculate_score_return_sorted_sellers(self):

        attr_weights = self.criteria_weights
        sellers = [a for a in self.schedule.agents[self.num_buyers-1:] if isinstance(a, Seller)]
        seller_scores = []

        for seller in sellers:
            score = 0
            for product in seller.products:
                weighted_product_attributes = [
                    product.seller_feedback * attr_weights[0],
                    product.search_relevance * attr_weights[1],
                    product.seller_response * attr_weights[2],
                    product.delivery_speed * attr_weights[3],
                    product.sales_history * attr_weights[4],
                    product.time_spent_on_product_page * attr_weights[5],
                    product.click_rate * attr_weights[6],
                    product.conversion * attr_weights[7],
                    product.views * attr_weights[8],
                    product.shopping_cart * attr_weights[9],
                    product.search_by_query_name * attr_weights[10],

                    ]
                score += sum(weighted_product_attributes)
            seller_scores.append(score)
            seller.score_for_sorting = score

        # sorted_sellersOLD = [s for _, s in sorted(zip(seller_scores, sellers), key=lambda pair: pair[0],
        #                                           reverse=True)]  # HIGHEST SCORE
        # sorted_sellers = [s for _, s in sorted(zip(seller_scores, sellers), key=lambda pair: pair[0])] # LOWEST SCORE FIRST
        sorted_sellers = [s for _, s in sorted(zip(seller_scores, sellers),
                                               key=lambda pair: pair[0] * (1 + random.uniform(-0.1, 0.1)),
                                               reverse=True)]  # HIGHEST WITH SOME RANDOMNESS

        return sorted_sellers

    def generate_random_values(self, n):
        random_values = [random.uniform(self.seller_feedback_lower, self.seller_feedback_upper),
                         random.uniform(self.search_relevance_lower, self.search_relevance_upper),
                         random.uniform(self.seller_response_lower, self.seller_response_upper),
                         random.uniform(self.delivery_speed_lower, self.delivery_speed_upper),
                         random.uniform(self.sales_history_lower, self.sales_history_upper)] + [random.random() for _ in range(n-5)]
        #random_values = [random.random() for _ in range(n)]
        return random_values


    def write_best_seller_to_file(self):
        sellers = [agent for agent in self.schedule.agents if isinstance(agent, Seller)]
        sorted_sellers = sorted(sellers, key=lambda seller: seller.sales, reverse=True)

        sellers_with_zero_sales = sum(seller.sales == 0 for seller in sellers)

        with open("buyer.txt", "a") as f:  # Change the mode to "a" for appending data
            f.write(f"\n--- Step {self.schedule.steps} ---\n")

            # Write the number of sellers with 0 sales during the 200 steps
            f.write(f"Sellers with 0 sales during the {NUMBER_STEPS_TO_WRITE} steps: {sellers_with_zero_sales}\n")

            # Write the weights
            f.write(f"seller_feedback: {self.criteria_weights[0]}\n")
            f.write(f"search_relevance: {self.criteria_weights[1]}\n")
            f.write(f"seller_reponse: {self.criteria_weights[2]}\n")
            f.write(f"delivery_speed: {self.criteria_weights[3]}\n")
            f.write(f"sales_history: {self.criteria_weights[4]}\n")
            f.write(f"time_spent_on_product_page: {self.criteria_weights[5]}\n")
            f.write(f"click_rate: {self.criteria_weights[6]}\n")
            f.write(f"conversion: {self.criteria_weights[7]}\n")
            f.write(f"views: {self.criteria_weights[8]}\n")
            f.write(f"shopping_cart: {self.criteria_weights[9]}\n")
            f.write(f"search_by_query_name: {self.criteria_weights[10]}\n")


            f.write(f"Missed sales: {self.missed_transactions} out of {self.missed_transactions + self.successful_transactions}")
            f.write(f"\nSuccessful sales: {self.successful_transactions}")
            gotten_data.append((sellers_with_zero_sales, self.successful_transactions, (self.criteria_weights[0],
                                                                                        self.criteria_weights[1],
                                                                                        self.criteria_weights[2],
                                                                                        self.criteria_weights[3],
                                                                                        self.criteria_weights[4],
                                                                                        self.criteria_weights[5],
                                                                                        self.criteria_weights[6],
                                                                                        self.criteria_weights[7],
                                                                                        self.criteria_weights[8],
                                                                                        self.criteria_weights[9],
                                                                                        self.criteria_weights[10])))


from mesa.visualization.modules import TextElement

class SalesDataElement(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        interactions_per_step = f"Interactions per Step: {model.interactions_per_step}"

        if model.schedule.steps < 200:
            return interactions_per_step

        seller_sales_data = model.datacollector.get_model_vars_dataframe()["Seller_Sales"].iloc[-1]

        result = "<br>".join(
            [f"{key}: {value}" for key, value in seller_sales_data.items()]
        )
        return f"Sales after 200 steps:<br>{result}<br>Successful Transactions: {model.successful_transactions}<br>Missed Transactions: {model.missed_transactions}<br>{interactions_per_step}"


def minkowski_distance(a, b, p=3):
    return sum(abs(x - y)**p for x, y in zip(a, b))**(1/p)

def minkowski_similarity(a, b, p=3):
    distance = minkowski_distance(a, b, p)
    # Add a small value to the denominator to avoid division by zero
    similarity = 1 / (1 + distance)
    return similarity

def write_string_to_file(filename, string):
    try:
        with open(filename, 'a') as file:
            file.write(string + '\n')
    except Exception as e:
        print(f"Error occurred: {e}")
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 0,
    }
    if isinstance(agent, Buyer):
        portrayal["Color"] = "blue"
    elif isinstance(agent, Seller):
        portrayal["Color"] = "green"
    else:
        portrayal["Color"] = "red"
    return portrayal


successful_sales_chart = ChartModule(
    [{"Label": "Successful_Sales", "Color": "blue"}],
    data_collector_name="datacollector",
)

grid = CanvasGrid(agent_portrayal, 110, 110, 450, 450)  # Assuming a 10x10 grid

chart = ChartModule(
    [{"Label": "Sales", "Color": "green"}],
    data_collector_name="datacollector",
)


model_params = {
    "n_buyers": UserSettableParameter("slider", "Number of Buyers", 4000, 1, 10000, 1),
    "n_sellers": UserSettableParameter("slider", "Number of Sellers", 50, 1, 100, 1),
    "n_success_threshold": UserSettableParameter("slider", "Success Threshold", 0.8, 0, 1, 0.01),
    "n_minimum_amount_products_view": UserSettableParameter("number", "Minimum products buyer views", 5),
    "seller_feedback_lower": UserSettableParameter("slider", "Buyers Attribute Seller Feedback Range Lower Bound", 0, 0, 1, 0.01),
    "seller_feedback_upper": UserSettableParameter("slider", "Buyers Attribute Seller Feedback Range Upper Bound", 1, 0, 1, 0.01),
    "search_relevance_lower": UserSettableParameter("slider", "Buyers Attribute Search Relevance Range Lower Bound", 0, 0, 1, 0.01),
    "search_relevance_upper": UserSettableParameter("slider", "Buyers Attribute Search Relevance Range Upper Bound", 1, 0, 1, 0.01),
    "seller_response_lower": UserSettableParameter("slider", "Buyers Attribute Seller Response Range Lower Bound", 0, 0, 1, 0.01),
    "seller_response_upper": UserSettableParameter("slider", "Buyers Attribute Seller Response Range Upper Bound", 1, 0, 1, 0.01),
    "delivery_speed_lower": UserSettableParameter("slider", "Buyers Attribute Delivery Speed Range Lower Bound", 0, 0, 1, 0.01),
    "delivery_speed_upper": UserSettableParameter("slider", "Buyers Attribute Delivery Speed Range Upper Bound", 1, 0, 1, 0.01),
    "sales_history_lower": UserSettableParameter("slider", "Buyers Attribute Sales History Range Lower Bound", 0, 0, 1, 0.01),
    "sales_history_upper": UserSettableParameter("slider", "Buyers Attribute Sales History Range Upper Bound", 1, 0, 1, 0.01),

    "width": 110,
    "height": 110,

}
total_buyers = int(model_params["n_buyers"].value)

total_sellers = int(model_params["n_sellers"].value)

sales_data = SalesDataElement()

server = ModularServer(
    OnlineShopModel,
    [grid],  # Add the sales_data module
    "Online Shop Simulation",
    model_params,
)

server.launch()







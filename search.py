import math
import operator
import random
import time

from tree import Tree


class Search:
    """
    A Class with many Artificial Intelligence Search Methods(eg:- bfs, dfs, a*, heuristics e.t.c).
    """

    def __init__(self, goal_test, next_states, state=None, heuristic=None):
        """
        :param state: The start state of the search problem.
        :param goal_test: The function that take state as input to test if the problem is at goal state.
        :param next_states: The function that takes state as input and returns the next possible states.
        """
        self.tree = None
        if state is not None:
            self.tree = Tree(root_nodes=[state], auto_correct=True)
            self.state = state
        self.algorithms = {
            'bfs': (self.bfs, ('verbose',)),
            'dfs': (self.dfs, ('verbose',)),
            'dls': (self.dls, ('depth', 'verbose', 'get_states',)),
            'dfids': (self.dfids, ('verbose',)),
            'best_fs': (self.best_fs, ('heuristic', 'verbose',)),
            'hill_climbing': (self.hill_climbing, ('heuristic', 'verbose', 'beam_width',)),
            'simulated_annealing': (self.simulated_annealing, ('heuristic', 'epochs', 'verbose', 'iterations',
                                                               'temperature', 'cooling',)),
        }
        self.goal_test = goal_test
        self.next_states = next_states
        self.heuristic = heuristic
        self.quit = False

    def set_state(self, state):
        """
        To set the start state of the search problem after initialising the Search class.

        :param state: The start state of the search problem.
        """
        self.state = state
        self.tree = Tree(root_nodes=[state], auto_correct=True)

    def non_visited_states(self, state) -> list:
        """
        Returns the list of the next non visited states for the given state.

        :param state: The state of the problem.
        """
        self.tree.add_children(state, self.next_states(state))
        return self.tree.get_children(state)

    def search(self, algorithm: str, show_time: bool = True, **kwargs) -> list:
        """
        Takes the name of the algorithm and solves the puzzle using that algorithm
        and returns the path from start state to the goal state.

        :param algorithm: The name of the algorithm as string.
        :param show_time: Displays the time taken by the algorithm to solve.
        :exception: Raises an Exception, if the type of the parameter 'algorithm' is not string.
        :exception: Raises an Exception, if the algorithm specified do not exist.
        """
        if type(algorithm) != str:
            raise Exception("type(algorithm) must be string.")
        try:
            args = {key: value for key, value in kwargs.items() if key in self.algorithms[algorithm][1]}
            start_time = time.time()
            solution = self.algorithms[algorithm][0](**args)
            time_taken = round(time.time() - start_time, 2)
            if show_time:
                print(f"Time Taken: {time_taken}")
            return solution
        except KeyError:
            raise Exception(f"No algorithm named {algorithm} found.")

    # Search Methods
    def bfs(self, verbose: bool = True) -> list:
        """
        Uses 'Breadth First Search(BFS)' algorithm to solve the problem.
        and returns the path from start state to goal state.

        :param verbose: Prints Output to the screen.
        :exception: Raises a Exception, if there is no solution for the given problem.
        :returns: The path from start state to goal state as a list.
        """
        all_states = []
        if verbose:
            print("**************Solving(BFS)*****************")
        depth_count = 0
        states = 1
        queue = [self.state]
        while len(queue) != 0:
            if verbose:
                print(f"\rDepth: {depth_count} | States: {states}", end='')
            new_open = []
            for state in queue:
                if self.quit:
                    quit()
                if self.goal_test(state):
                    if verbose:
                        print()
                    return self.tree.get_path(state)
                new_open += self.non_visited_states(state)
            queue = new_open
            depth_count += 1
            states += len(queue)
            all_states.extend(queue)
        raise StopIteration("Can't find Solution.")

    def dfs(self, verbose: bool = True) -> list:
        """
        Uses 'Depth First Search(DFS)' algorithm to solve the problem
        and returns the path from start state to goal state.

        :param verbose: Prints Output to the screen.
        :type verbose: bool
        :exception: Raises a Exception, if there is no solution for the given problem.
        :returns: The path from start state to goal state as a list.
        """
        if verbose:
            print("**************Solving(DFS)*****************")
        depth_count = 0
        states = 1
        stack = [self.state]
        while len(stack) != 0:
            if verbose:
                print(f"\rDepth: {depth_count} | States: {states}", end='')
            if self.quit:
                quit()
            state = stack.pop()
            if self.goal_test(state):
                if verbose:
                    print()
                return self.tree.get_path(state)
            nvs = self.non_visited_states(state)
            if len(nvs) == 0:
                self.tree.delete(state)
                depth_count -= 1
            stack += nvs
            self.tree.add_children(state, nvs)
            depth_count += 1
            states += len(nvs)
        raise Exception("Can't find Solution.")

    def dls(self, depth: int = 0, verbose: bool = True, get_sates: bool = False) -> [list, int]:
        """
        Uses 'Depth Limited Search(DLS)' algorithm to solve the problem.
        and returns the path from start state to goal state.

        :param depth: The depth_limit to search.
        :param verbose: Prints Output to the screen.
        :param get_sates: Returns the number of states instead fo raising Exception.
        :exception: Raises a Exception, if there is no solution for the given problem at specified depth.
        :returns: The path from start state to goal state as a list.
        """
        if verbose:
            print("**************Solving(DLS)*****************")
        stack = [self.state]
        states = 1
        while len(stack) != 0:
            if self.quit:
                quit()
            state = stack.pop()
            state_depth = self.tree.get_depth(state)
            if self.goal_test(state):
                print()
                return self.tree.get_path(state)
            if state_depth <= depth:
                if verbose:
                    print(f"\rDepth: {state_depth} | States: {states}", end='')
                nvs = self.non_visited_states(state)
                if len(nvs) == 0:
                    self.tree.delete(state)
                    pass
                self.tree.add_children(state, nvs)
                stack += nvs
                states += len(nvs)
        if get_sates:
            return states
        raise Exception("Can't find Solution in the specified depth try increasing depth.")

    def dfids(self, verbose: bool = True) -> list:
        """
        Uses 'Depth First Iterative Deepening Search(DFIDS)' algorithm to solve the problem
        and returns the path from start state to goal state.

        :param verbose: Prints Output to the screen.
        :warning: Continues in a infinite loop if No Solution exists for the problem.
        :returns: The path from start state to goal state as a list.
        """
        if verbose:
            print("**************Solving(DFIDS)*****************")
        depth_count = 0
        states = 1
        while True:
            if verbose:
                print(f"\rIteration: {depth_count} | States: {states}", end='')
            if self.quit:
                quit()
            solution = self.dls(depth=depth_count, verbose=False, get_sates=True)
            if type(solution) == list:
                return solution
            else:
                states += solution
            depth_count += 1

    def best_fs(self, heuristic=None, verbose: bool = True) -> list:
        """
        Uses 'Best First Search(BEST_FS)' algorithm to solve the problem.
        and returns the path from start state to goal state.

        :param heuristic: A function to calculate the heuristic of a given state.
                          (The better the state the greater should be the heuristic value)
        :param verbose: Prints Output to the screen.
        :exception: Raises a Exception, if there is no solution for the given problem.
        :exception: Raises a Exception, if no heuristic function is provided.
        :returns: The path from start state to goal state as a list.
        """
        heuristic = self.heuristic if heuristic is None else heuristic
        if heuristic is None:
            raise Exception("No heuristic function is provided.")
        if verbose:
            print("**************Solving(BEST_FS)*****************")
        depth_count = 0
        states = 1
        queue = [self.state]
        while len(queue) != 0:
            current_state = queue.pop()
            if verbose:
                print(f"\rDepth: {depth_count} | States: {states}", end='')
            if self.quit:
                quit()
            if self.goal_test(current_state):
                if verbose:
                    print()
                return self.tree.get_path(current_state)
            next_states = self.non_visited_states(current_state)
            heuristics = [heuristic(state) for state in next_states]
            queue.extend([state for _, state in sorted(zip(heuristics, next_states))])
            depth_count += 1
            states += len(next_states)
        raise Exception("Can't find Solution.")

    def hill_climbing(self, heuristic=None, beam_width: int = 1, verbose: bool = True) -> list:
        """
        Uses 'Hill Climbing' algorithm to optimize the problem.
        and returns the optimised solution.

        :param heuristic: A function to calculate the heuristic of a given state.
                          (The better the state the greater should be the heuristic value)
        :param beam_width: The number of top states to extend.
        :param verbose: Prints Output to the screen.
        :exception: Raises a Exception, if no heuristic function is provided.
        :returns: The optimised solution(May not be the global optimum).
        """
        heuristic = self.heuristic if heuristic is None else heuristic
        depth_count = 0
        states = 1
        best_state = self.state
        current_states = [self.state]

        def top_moves(problem_state, top=beam_width):
            neighbour_states = self.non_visited_states(problem_state)
            neighbour_heuristics = [heuristic(the_state) for the_state in neighbour_states]
            ordered_list = [the_state for _, the_state in sorted(zip(neighbour_heuristics, neighbour_states))]
            if len(ordered_list) < top:
                return ordered_list
            else:
                for extra_state in ordered_list[:-1 * top]:
                    self.tree.delete(extra_state)
                return ordered_list[-1 * top:]

        if heuristic is None:
            raise Exception("No heuristic function is provided.")
        if verbose:
            print("**************Solving(HILL_CLIMBING)*****************")

        while True:
            if verbose:
                print(f"\rDepth: {depth_count} | States: {states}", end='')
            if self.quit:
                quit()
            for state in current_states:
                if self.goal_test(state):
                    if verbose:
                        print()
                    return self.tree.get_path(state)
            try:
                new_neighbours = []
                for state in current_states:
                    new_neighbours.extend(top_moves(state))
                    depth_count += 1
                    best_state = new_neighbours[-1] if heuristic(new_neighbours[-1]) > heuristic(best_state) \
                        else best_state
                states += len(new_neighbours)
                current_states = new_neighbours
            except IndexError:
                break
        if verbose:
            print()
        return self.tree.get_path(best_state)

    def simulated_annealing(self, heuristic=None, temperature: float = 1000, cooling: float = 0.9,
                            epochs: int = 1000, iterations: int = 1000, verbose: bool = True) -> list:
        """
        Uses Simulated annealing to optimize the solution.
        
        :param heuristic: 
        :param temperature: 
        :param cooling: 
        :param epochs: 
        :param iterations: 
        :param verbose: 
        :return: 
        """
        heuristic = self.heuristic if heuristic is None else heuristic
        current_state = self.state
        best_state = self.state
        temperature = temperature
        if heuristic is None:
            raise Exception("No heuristic function is provided.")
        if verbose:
            print("**************Optimizing(SIMULATED_ANNEALING)*****************")
        for epoch in range(epochs):
            for iteration in range(iterations):
                if verbose:
                    print(f"\rEpoch: {epoch} | Iteration: {iteration}", end="")
                if self.quit:
                    quit()
                try:
                    random_state = random.choice(self.non_visited_states(current_state))
                    delta_eval = heuristic(random_state) - heuristic(current_state)
                    if random.uniform(0, 1) < (1 / (1 + math.e ** ((-1 * delta_eval) / temperature))):
                        current_state = random_state
                        if heuristic(current_state) >= heuristic(best_state):
                            best_state = current_state
                except IndexError:
                    pass
            temperature *= cooling
            current_state = self.state
            # self.reset([current_state])

        if verbose:
            print("Optimized")
        return best_state

    def genetic_algorithm(self, population: list, fitness, crossover, mutate=None, mutate_percent: float = 0.2,
                          k=None, epochs: int = None, verbose: bool = True):
        if k is None:
            k = len(population)
        elif k > len(population):
            raise ValueError("parameter 'k' should be less than or equal to len(population)")

        epochs_count = 0
        best_state = population[0]
        if verbose:
            print("**************Optimizing(GENETIC ALGORITHM)*****************")
        while True:
            # Checking for solution and computing the fitness values and updating the best state
            fitness_values = []
            best_fitness = fitness(best_state)
            for genome in population:
                fitness_value = fitness(genome)
                # Updating the best Fitness
                if fitness_value > best_fitness:
                    best_state = genome
                    best_fitness = fitness(best_state)
                fitness_values.append(fitness_value)
                if verbose:
                    print(
                        f"\rEpoch: {epochs_count} | Best Fitness: {fitness(best_state)} | Best Value: {fitness(best_state)}",
                        end='')
                # Checking for termination criteria
                if self.goal_test is not None:
                    if self.goal_test(genome):
                        return genome

            # Selecting population based on their fitness
            len_population = len(population)
            total_fitness = sum(fitness_values)
            selected_population = []
            for i in range(len_population):
                for _ in range(math.ceil((fitness_values[i] / total_fitness) * len_population)):
                    selected_population.append(population[i])
            selected_population = selected_population[:len_population]

            # Splitting the population into to half's(parent1-male, parent2-female)
            new_population = []
            male, female = selected_population[:int(len_population / 2)], selected_population[int(len_population / 2):]
            if len_population % 2 != 0:
                if len(male) % 2 != 0:
                    new_population.append(male.pop(0))
                elif len(female) % 2 != 0:
                    new_population.append(female.pop(0))

            # Performing crossover on parents 1, 2 to generate 2 children and appending them to the new_population
            for parent1, parent2 in zip(male, female):
                new_population.extend(crossover(parent1, parent2))

            # Mutating mutate_percentage of new_population
            if mutate is not None:
                mutated_indexes = []
                for _ in range(math.ceil(mutate_percent * len_population)):
                    rand_int = random.randint(0, len_population - 1)
                    if rand_int not in mutated_indexes:
                        mutated_indexes.append(rand_int)
                        new_population[rand_int] = mutate(new_population[rand_int])

            # Updating weakest k genomes in the population with strongest k genomes in the new_population
            if k == len_population:
                population = new_population
            else:
                sorted_population = [genome for _, genome in sorted(zip(fitness_values, population),
                                                                    key=operator.itemgetter(0))]
                new_population_fitness = [fitness(genome) for genome in new_population]
                sorted_new_population = [genome for _, genome in sorted(zip(new_population_fitness,
                                                                            new_population), reverse=True,
                                                                        key=operator.itemgetter(0))]
                population = sorted_population
                population[:k] = sorted_new_population[:k]

            if epochs_count == epochs:
                return best_state
            epochs_count += 1


# Example-1
# if __name__ == '__main__':
#     print("--------------------------------Problem-1--------------------------------")
#     search = Search(goal_test=lambda state: state == 10,
#                     next_states=lambda state: [state + 1, state + 2, state + 3], state=0)
#     path = search.search(algorithm='dfids')
#     print("Path:", path)
#
# # Example-2
# if __name__ == '__main__':
#     print("--------------------------------Problem-2--------------------------------")
#     graph = Tree(['a'])
#     graph.add_children('a', ['b', 'c', 'd'])
#     graph.add_children('b', ['e', 'f'])
#     graph.add_children('c', ['g'])
#     graph.add_children('d', ['h', 'i'])
#     graph.add_children('e', ['l', 'm'])
#     graph.add_children('i', ['j', 'k'])
#     search = Search(goal_test=lambda state: state == 'j',
#                     next_states=graph.get_children, state='a')
#     path = search.search(algorithm='bfs')
#     print("Path:", path)
#
# # Example-3
# if __name__ == '__main__':
#     print("--------------------------------Problem-3--------------------------------")
#     present_state = (0, 1, 2)
#
#
#     def state_value(state):
#         return - 1 / sum(state)
#
#
#     def neighbours(state):
#         next_states = []
#         if state_value(state) < 101:
#             for i in range(1, 4):
#                 next_states.append((state[0] + i, state[1] + i, state[2] + i))
#             return next_states
#         elif state_value(state) < 201:
#             for i in range(1, 4):
#                 next_states.append((state[0] - i, state[1] - i, state[2] - i))
#             return next_states
#         else:
#             for i in range(1, 4):
#                 next_states.append((state[0] + i + 100, state[1] + i + 100, state[2] + i + 100))
#             return next_states
#
#
#     search = Search(goal_test=lambda state: state == (45, 46, 47),
#                     next_states=neighbours,
#                     state=present_state)
#     print(search.search('simulated_annealing', heuristic=state_value, epochs=100, iterations=10000))
#     print(search.search('hill_climbing', heuristic=state_value))

# Example-4
if __name__ == '__main__':
    def function(genome):
        return (1.223*genome[0]/4 + 9/genome[1]*2 + 12*genome[2]/6) - 2500


    def is_goal(state):
        if 0 <= abs(function(state)) <= 0.1:
            return True
        return False


    def mutate(genome):
        a, b = 0.9, 1.1
        var1 = genome[0] * random.uniform(a, b)
        var2 = genome[1] * random.uniform(a, b)
        var3 = genome[2] * random.uniform(a, b)
        return var1, var2, var3


    def cross_over(parent1, parent2):
        mix = list(parent1) + list(parent2)
        random.shuffle(mix)
        return [mix[:3], mix[3:]]


    def score(genome):
        answer = function(genome)
        if answer == 0:
            return 99999
        else:
            return abs(1 / answer)


    rand_population = [(random.uniform(1, 10000), random.uniform(1, 10000), random.uniform(1, 10000))
                       for _ in range(1000)]
    search = Search(state=(1000, 100, 100), goal_test=is_goal, next_states=None)
    output = search.genetic_algorithm(population=rand_population, fitness=score, crossover=cross_over, epochs=10000,
                             mutate=mutate, mutate_percent=0.005, k=300)
    print(output)

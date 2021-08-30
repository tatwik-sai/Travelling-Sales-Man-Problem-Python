import pygame
from itertools import permutations
import math
import threading

pygame.init()


class TSP:
    """
    Travelling Salesman Problem(TSP) Visualiser.
    """

    def __init__(self):
        # Variables
        self.running = True
        self.min_distance = 50
        self.points = []
        self.center_points = []
        self.distances = {}
        self.lines = []
        self.num_points = 0

        self.stop_solving = False
        self.solving = False
        self.total_states = 0
        self.searched_states = 0
        self.progress = "0.00"
        self.best_dst = "N/A"

        # Display
        self.screen_size = (1366, 700)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("TSP")
        pygame.display.set_icon(pygame.image.load('graph-icon.png'))
        self.point_size = 20

        # Colors
        self.bg_color = (39, 38, 39)
        self.point_color = (255, 255, 255)
        self.line_color = (93, 90, 92)
        self.title_color = (255, 128, 0)
        self.subheading_color = (150, 142, 140)
        self.info_color = (94, 89, 89)
        self.solve_button_color = (255, 255, 255)
        self.reset_button_color = (255, 255, 255)

        # Info Display
        self.info_rect_dims = (int(self.screen_size[0] / 3), int(self.screen_size[1] / 3))
        self.info_rect = pygame.Rect((0, 0), self.info_rect_dims)

        # Fonts
        self.solve_font = pygame.font.SysFont('georgia', 25)
        self.title_font = pygame.font.SysFont('microsofthimalaya', 40)
        self.sub_heading_font = pygame.font.SysFont('javanesetext', 20)
        self.info_font = pygame.font.SysFont('microsofthimalaya', 30)

        # Texts
        self.searched_text = self.sub_heading_font.render("Searched: ", True, self.subheading_color)
        self.progress_text = self.sub_heading_font.render("Progress: ", True, self.subheading_color)
        self.bstdst_text = self.sub_heading_font.render("Best dst: ", True, self.subheading_color)

        self.solve_text = self.solve_font.render("Solve", True, self.info_color)
        self.reset_text = self.solve_font.render("Reset", True, self.info_color)

        # Solve Button
        self.solve_on_hold = False
        s_but_x, s_but_y = self.info_rect_dims[0], self.info_rect_dims[1]
        s_button_pos = (int(s_but_x - s_but_x * 0.98), int(s_but_y - s_but_y / 7))
        self.solve_button = pygame.Rect(s_button_pos, (80, 30))
        self.solve_text_pos = (s_button_pos[0] + 10, s_button_pos[1] + 3)

        # Reset Button
        self.reset_on_hold = False
        r_button_pos = (s_button_pos[0] + 100, s_button_pos[1])
        self.reset_button = pygame.Rect(r_button_pos, (80, 30))
        self.reset_text_pos = (r_button_pos[0] + 10, r_button_pos[1] + 3)

    @staticmethod
    def distance(p1: tuple, p2: tuple) -> float:
        """
        Returns the sqrt of the squared distance between two points.

        :param p1: Coordinates of point1.
        :param p2: Coordinates of point2.
        :return: The Distance between point1 and point2.
        """
        return round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 2)

    def compute_distances(self):
        """
        Updates distances dictionary with distances from each point to every other point.
        """
        self.distances.clear()
        num_points = len(self.points)
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    pass
                else:
                    self.distances[str(i) + "-" + str(j)] = self.distance(self.points[i], self.points[j])

    def reset_action(self):
        """
        Resets the program when reset button is clicked.
        """
        if self.solving:
            self.stop_solving = True
        self.reset_on_hold = False
        self.reset_button_color = (255, 255, 255)
        self.points.clear(), self.center_points.clear(), self.lines.clear()
        self.num_points = 0
        self.total_states = 0
        self.searched_states = 0
        self.progress = "0.00"
        self.best_dst = "N/A"

    def solve_action(self):
        """
        Calls the solve function in a new thread to solve the TSP problem.
        """
        if not self.solving:
            self.solve_on_hold = False
            self.solve_button_color = (237, 204, 100)
            self.compute_distances()
            threading.Thread(target=self.brute_force_solve).start()

    def add_point(self, pos: tuple):
        """
        Adds a point on the clicked position if no point is there in 50pixel radius around it.

        :param pos: The (x, y) coordinates of the point.
        """
        self.total_states = 0
        self.searched_states = 0
        self.progress = "0.00"
        self.best_dst = "N/A"
        for point in self.points:
            if self.distance((point.x, point.y), pos) < self.min_distance:
                break
        else:
            self.num_points += 1
            self.lines.clear()
            self.points.append(pygame.Rect(pos, (self.point_size, self.point_size)))
            self.center_points.append((int(pos[0] + self.point_size / 2), (int(pos[1] + self.point_size / 2))))
            self.total_states = 0 if self.num_points == 1 else int(math.factorial(self.num_points - 1) / 2)

    def remove_point(self, pos: tuple):
        """
        Removes a point from the clicked position if a point exists in that position.

        :param pos: The (x, y) coordinates of the point.
        """
        self.total_states = 0
        self.searched_states = 0
        self.progress = "0.00"
        self.best_dst = "N/A"
        self.total_states = 0 if self.num_points == 1 else int(math.factorial(self.num_points - 1) / 2)
        for i in range(len(self.points)):
            if self.points[i].collidepoint(pos):
                self.num_points -= 1
                self.title_text = self.title_font.render(f"Solving {self.num_points} point problem", True,
                                                         self.title_color)
                self.lines.clear()
                self.points.pop(i)
                self.center_points.pop(i)
                break

    def handle_click(self, event):
        """
        Handles Mouse clicks to add or remove points (or) to solve or reset the program.

        :param event: pygame.event object.
        """
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            if event.button == 1:
                if self.solve_button.collidepoint(pos):
                    self.solve_action()
                elif self.reset_button.collidepoint(pos):
                    self.reset_action()
                elif self.info_rect.collidepoint(pos):
                    pass
                elif pos[0] > self.screen_size[0] - self.point_size or pos[1] > self.screen_size[1] - self.point_size:
                    pass
                elif not self.solving:
                    self.add_point(pos)
            elif event.button == 3 and not self.solving:
                self.remove_point(pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                if self.solve_button.collidepoint(pos) and not self.solving:
                    self.solve_on_hold = True
                elif self.reset_button.collidepoint(pos):
                    self.reset_on_hold = True

    def draw_text(self, text: str, font: pygame.font, color: tuple, pos: tuple):
        """
        Draws text on to the screen.

        :param text: The text to be drawn.
        :param font: pygame.font Object
        :param color: The color of the text.
        :param pos: The position of the text.
        """
        self.screen.blit(font.render(text, True, color), pos)

    def draw(self):
        """
        Draws every thing on to the screen.
        """
        self.screen.fill(self.bg_color)

        # Setting Button Color
        if self.solve_on_hold:
            self.solve_button_color = (237, 204, 100)
        if self.reset_on_hold:
            self.reset_button_color = (237, 204, 100)

        # Information Rectangle
        pygame.draw.rect(self.screen, self.bg_color, self.info_rect)

        # Lines
        for line in self.lines:
            pygame.draw.aaline(self.screen, self.line_color, line[0], line[1])

        # Points
        for point in self.points:
            pygame.draw.rect(self.screen, self.point_color, point, border_radius=50)

        # Drawing Button Rectangles
        pygame.draw.rect(self.screen, self.solve_button_color, self.solve_button, border_radius=5)
        pygame.draw.rect(self.screen, self.reset_button_color, self.reset_button, border_radius=5)

        # Button Text
        self.screen.blit(self.solve_text, self.solve_text_pos)
        self.screen.blit(self.reset_text, self.reset_text_pos)

        # Title Text
        self.draw_text(f"Solving {self.num_points} point problem", self.title_font, self.title_color, (10, 10))

        # Subheading Text
        self.screen.blit(self.searched_text, (10, 40))
        self.screen.blit(self.progress_text, (10, 70))
        self.screen.blit(self.bstdst_text, (10, 100))

        # Info Text
        self.draw_text(f"{self.searched_states} / {self.total_states}", self.info_font, self.info_color, (100, 48))
        self.draw_text(f"{self.progress} %", self.info_font, self.info_color, (100, 78))
        self.draw_text(str(self.best_dst), self.info_font, self.info_color, (100, 108))

        pygame.display.update()

    def cost(self, path: tuple) -> float:
        """
        Returns the cost of the tour.
        :param path: The path of the tour.
        :return: The distance of the complete tour.
        """
        cost = 0
        for i in range(self.num_points):
            end = i + 1 if i < self.num_points - 1 else 0
            cost += self.distances[path[i] + "-" + path[end]]
        return cost

    def construct_path(self, path):
        """
        Adds the lines based on the path.
        :param path:  The path of the tour.
        """
        self.lines.clear()
        for i in range(self.num_points):
            end = i + 1 if i < self.num_points - 1 else 0
            self.lines.append([self.center_points[int(path[i])], self.center_points[int(path[end])]])

    def brute_force_solve(self):
        """
        Uses BruteForce approach to solve.
        """
        self.solving = True
        points = [str(i) for i in range(self.num_points)]
        try:
            if self.num_points > 1:
                self.searched_states = 0
                self.best_dst = 10e10000

                for path in permutations(points):
                    cost = self.cost(path)

                    if cost < self.best_dst:
                        self.best_dst = round(cost, 2)
                        self.construct_path(path)

                    if self.searched_states == self.total_states or self.stop_solving:
                        self.stop_solving = False
                        self.progress = 100.00
                        break
                    self.searched_states += 1
        except TypeError:
            pass
        finally:
            self.solve_button_color = (255, 255, 255)
            self.solving = False

    def main(self):
        """
        Main function to handle the pygame loop.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_solving = True
                    self.running = False
                    pygame.quit()
                self.handle_click(event)

            if not self.running:
                break
            if self.solving:
                try:
                    self.progress = round((100 * self.searched_states) / self.total_states, 2)
                except ZeroDivisionError:
                    pass
            self.draw()


tsp = TSP()
tsp.main()

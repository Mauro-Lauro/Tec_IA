import pygame
import heapq

pygame.init()

WINDOW_WIDTH = 800  
GRID_SIZE = 5     
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
pygame.display.set_caption("Asterisco")

BACKGROUND = (10, 10, 20)     
GRID_LINES = (30, 30, 50)     
START_NODE = (0, 200, 255)    
END_NODE = (255, 80, 0)       
PATH_COLOR = (0, 255, 180)    
WALL_COLOR = (40, 40, 80)     
VISITED_COLOR = (120, 0, 255) 
NODE_COLOR = (20, 20, 40)     

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = BACKGROUND
        self.width = width
        self.total_rows = total_rows
        
    def get_pos(self):
        return self.row, self.col
        
    def is_wall(self):
        return self.color == WALL_COLOR
        
    def reset(self):
        self.color = BACKGROUND
        
    def make_start(self):
        self.color = START_NODE
        
    def make_wall(self):
        self.color = WALL_COLOR
        
    def make_end(self):
        self.color = END_NODE
        
    def make_path(self):
        self.color = PATH_COLOR
        
    def make_visited(self):
        self.color = VISITED_COLOR
        
    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.width - 2, self.width - 2), border_radius=3)

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def draw_grid_lines(window, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(window, GRID_LINES, (0, i * gap), (width, i * gap), 1)
        for j in range(rows):
            pygame.draw.line(window, GRID_LINES, (j * gap, 0), (j * gap, width), 1)

def draw_interface(window, grid, rows, width):
    window.fill(BACKGROUND)
    for row in grid:
        for node in row:
            node.draw(window)
    draw_grid_lines(window, rows, width)
    pygame.display.update()

def get_clicked_position(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def get_neighbors(grid, node):
    neighbors = []
    row, col = node.get_pos()
    
    directions = [
        (-1, 0, 1),      
        (1, 0, 1),       
        (0, -1, 1),      
        (0, 1, 1),       
        (-1, -1, 1.414), 
        (-1, 1, 1.414),  
        (1, -1, 1.414),  
        (1, 1, 1.414)    
    ]
    
    for dr, dc, cost in directions:
        r, c = row + dr, col + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and not grid[r][c].is_wall():
            neighbors.append((grid[r][c], cost))
    
    return neighbors

def astar(draw, grid, start, end):
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start))
    came_from = {}
    
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while open_set:
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)
        
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
            
        for neighbor, cost in get_neighbors(grid, current):
            temp_g_score = g_score[current] + cost
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_visited()
        
        draw()
        
        if current != start:
            current.make_visited()
    
    return False

def main(window, rows, width):
    grid = make_grid(rows, width)
    start = None
    end = None
    running = True
    
    while running:
        draw_interface(window, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                
            if pygame.mouse.get_pressed()[0]: 
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, rows, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != start and node != end:
                    node.make_wall()
                    
            elif pygame.mouse.get_pressed()[2]:  
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, rows, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
                    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and start and end:
                    astar(lambda: draw_interface(window, grid, rows, width), grid, start, end)
                    
    pygame.quit()

if __name__ == "__main__":
    main(WINDOW, GRID_SIZE, WINDOW_WIDTH)
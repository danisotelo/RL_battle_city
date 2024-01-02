import heapq
import random
import time
import multiprocessing
import pygame
import math
import queue


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class ai_agent():
    mapinfo = []
    # castle rect
    castle_rect = pygame.Rect(12 * 16, 24 * 16, 32, 32)

    def __init__(self):
        self.mapinfo = []

    # rect:                   [left, top, width, height]
    # rect_type:              0:empty 1:brick 2:steel 3:water 4:grass 5:froze
    # castle_rect:            [12*16, 24*16, 32, 32]
    # mapinfo[0]:             bullets [rect, direction, speed]]
    # mapinfo[1]:             enemies [rect, direction, speed, type]]
    # enemy_type:             0:TYPE_BASIC 1:TYPE_FAST 2:TYPE_POWER 3:TYPE_ARMOR
    # mapinfo[2]:             tile     [rect, type] (empty don't be stored to mapinfo[2])
    # mapinfo[3]:             player     [rect, direction, speed, Is_shielded]]
    # shoot:                  0:none 1:shoot
    # move_dir:               0:Up 1:Right 2:Down 3:Left 4:None

    # def Get_mapInfo:        fetch the map infomation
    # def Update_Strategy     Update your strategy

    def operations(self, p_mapinfo, c_control):

        while True:
            # -----your ai operation,This code is a random strategy,please design your ai !!-----------------------
            self.Get_mapInfo(p_mapinfo)

            player_rect = self.mapinfo[3][0][0]
            # sort enemy with manhattan distance to castle

            sorted_enemy_with_distance_to_castle = sorted(self.mapinfo[1],
                                                          key=lambda x: self.manhattan_distance(x[0].center,
                                                                                                self.castle_rect.center))
            # sort enemy with manhattan distance to player current position
            sorted_enemy_with_distance_to_player = sorted(self.mapinfo[1],
                                                          key=lambda x: self.manhattan_distance(x[0].center,
                                                                                                player_rect.center))

            # default position
            default_pos_rect = pygame.Rect(195, 3, 26, 26)
            # exists enemy
            if sorted_enemy_with_distance_to_castle:
                # if enemy distance with castle < 150, chase it
                if self.manhattan_distance(sorted_enemy_with_distance_to_castle[0][0].topleft, self.castle_rect.topleft) < 150:
                    enemy_rect = sorted_enemy_with_distance_to_castle[0][0]
                    enemy_direction = sorted_enemy_with_distance_to_castle[0][1]
                # else chase the nearest enemy to player
                else:
                    enemy_rect = sorted_enemy_with_distance_to_player[0][0]
                    enemy_direction = sorted_enemy_with_distance_to_player[0][1]

                # check if inline with enemy
                inline_direction = self.inline_with_enemy(player_rect, enemy_rect)

                # perform a star
                astar_direction = self.a_star(player_rect, enemy_rect, 6)

                # perform bullet avoidance
                shoot, direction = self.bullet_avoidance(self.mapinfo[3][0], 6, self.mapinfo[0], astar_direction, inline_direction)
                #print(shoot, direction)

                # update strategy
                self.Update_Strategy(c_control, shoot, direction)
                time.sleep(0.005)

            # go to default position
            else:
                # perform a star
                astar_direction = self.a_star(player_rect, default_pos_rect, 6)

                # update strategy
                if astar_direction is not None:
                    self.Update_Strategy(c_control, 0, astar_direction)
                    # time.sleep(0.001)
                else:
                    self.Update_Strategy(c_control, 0, 0)
                    # time.sleep(0.001)

            # ------------------------------------------------------------------------------------------------------

    def Get_mapInfo(self, p_mapinfo):
        if p_mapinfo.empty() != True:
            try:
                self.mapinfo = p_mapinfo.get(False)
            except queue.Empty:
                skip_this = True

    def Update_Strategy(self, c_control, shoot, move_dir):
        if c_control.empty() == True:
            c_control.put([shoot, move_dir])

    def should_fire(self, player_rect, enemy_rect_info_list):
        for enemy_rect_info in enemy_rect_info_list:
            if self.inline_with_enemy(player_rect, enemy_rect_info[0]) is not False:
                return True

    # A* algorithm, return a series of command to reach enemy
    def a_star(self, start_rect, goal_rect, speed):
        # print 'trigger a*'
        start = (start_rect.left, start_rect.top)
        goal = (goal_rect.left, goal_rect.top)

        # initialise frontier
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        # put start into frontier
        frontier.put(start, 0)
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current_left, current_top = frontier.get()
            current = (current_left, current_top)

            # goal test
            temp_rect = pygame.Rect(current_left, current_top, 26, 26)
            if self.is_goal(temp_rect, goal_rect):
                break

            # try every neighbour
            for next in self.find_neighbour(current_top, current_left, speed, goal_rect):
                # calculate new cost
                new_cost = cost_so_far[current] + speed

                # update if next haven't visited or cost more
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        # build path
        # dir_cmd = []
        # while current != start:
        #     parent = came_from[current]
        #     parent_left, parent_top = parent
        #     current_left, current_top = current
        #     # up
        #     if current_top < parent_top:
        #         dir_cmd.append(0)
        #     # down
        #     elif current_top > parent_top:
        #         dir_cmd.append(2)
        #     # left
        #     elif current_left < parent_left:
        #         dir_cmd.append(3)
        #     # right
        #     elif current_left > parent_left:
        #         dir_cmd.append(1)
        #     current = came_from[current]
        # dir_cmd.reverse()

        # return the first move is enough
        next = None
        dir_cmd = None
        while current != start:
            next = current
            current = came_from[current]

        if next:
            next_left, next_top = next
            current_left, current_top = current
            # up
            if current_top > next_top:
                dir_cmd = 0
            # down
            elif current_top < next_top:
                dir_cmd = 2
            # left
            elif current_left > next_left:
                dir_cmd = 3
            # right
            elif current_left < next_left:
                dir_cmd = 1
        return dir_cmd

    def manhattan_distance(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # heuristic func, use euclidean dist
    def heuristic(self, a, b):
        return self.manhattan_distance(a, b)

    # return True when two rects collide
    def is_goal(self, rect1, rect2):
        center_x1, center_y1 = rect1.center
        center_x2, center_y2 = rect2.center
        if abs(center_x1 - center_x2) <= 7 and abs(center_y1 - center_y2) <= 7:
            return True
        else:
            return False


    # return [(top,left)]
    # each time move 2px (speed)
    def find_neighbour(self, top, left, speed, goal_rect):

        # Rect(left, top, width, height)
        allowable_move = []

        # move up
        new_top = top - speed
        new_left = left
        if not (new_top < 0):
            move_up = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # check collision with enemy except goal
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_up = False
                        break

            # check collision with bullet
            # for bullet in self.mapinfo[0]:
            #     if temp_rect.colliderect(bullet[0]):
            #         move_up = False
            #         break

            # check collision with tile
            if move_up:
                for tile in self.mapinfo[2]:
                    # not a grass tile
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_up = False
                            break

            if move_up:
                allowable_move.append((new_left, new_top))

        # move right
        new_top = top
        new_left = left + speed
        if not (new_left > (416 - 26)):
            move_right = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # check collision with enemy except goal
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_right = False
                        break

            # check collision with bullet
            # for bullet in self.mapinfo[0]:
            #     if temp_rect.colliderect(bullet[0]):
            #         move_right = False
            #         break

            # check collision with tile
            if move_right:
                for tile in self.mapinfo[2]:
                    # not a grass tile
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_right = False
                            break

            if move_right:
                allowable_move.append((new_left, new_top))

        # move down
        new_top = top + speed
        new_left = left
        if not (new_top > (416 - 26)):
            move_down = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # check collision with enemy except goal
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_down = False
                        break

            # check collision with bullet
            # for bullet in self.mapinfo[0]:
            #     if temp_rect.colliderect(bullet[0]):
            #         move_down = False
            #         break

            # check collision with
            if move_down:
                for tile in self.mapinfo[2]:
                    # not a grass tile
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_down = False
                            break

            if move_down:
                allowable_move.append((new_left, new_top))

        # move left
        new_top = top
        new_left = left - speed
        if not (new_left < 0):
            move_left = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # check collision with enemy except goal
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_left = False
                        break

            # check collision with bullet
            # for bullet in self.mapinfo[0]:
            #     if temp_rect.colliderect(bullet[0]):
            #         move_left = False
            #         break

            # check collision with tile
            if move_left:
                for tile in self.mapinfo[2]:
                    # not a grass tile
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_left = False
                            break

            if move_left:
                allowable_move.append((new_left, new_top))

        return allowable_move

    def inline_with_enemy(self, player_rect, enemy_rect):
        # vertical inline
        if enemy_rect.left <= player_rect.centerx <= enemy_rect.right and abs(player_rect.top - enemy_rect.bottom) <= 151:
            # enemy on top
            if enemy_rect.bottom <= player_rect.top:
                #print('enemy on top')
                return 0
            # enemy on bottom
            elif player_rect.bottom <= enemy_rect.top:
                #print('enemy on bottom')
                return 2
        # horizontal inline
        if enemy_rect.top <= player_rect.centery <= enemy_rect.bottom and abs(player_rect.left - enemy_rect.right) <= 151:
            # enemy on left
            if enemy_rect.right <= player_rect.left:
                #print('enemy on left')
                return 3
            # enemy on right
            elif player_rect.right <= enemy_rect.left:
                #print('enemy on right')
                return 1
        return False

    def bullet_avoidance(self, player_info, speed, bullet_info_list, direction_from_astar, inlined_with_enemy):
        # possible direction list
        directions = []

        # player rect
        player_rect = player_info[0]

        # sort bullet by euclidean distance with player
        sorted_bullet_info_list = sorted(bullet_info_list, key=lambda x: self.euclidean_distance((x[0].left, x[0].top), (player_rect.centerx, player_rect.centery)))

        # default shoot
        shoot = 0

        # default minimal distance with bullet, infinity
        if sorted_bullet_info_list:
            min_dist_with_bullet = self.euclidean_distance((sorted_bullet_info_list[0][0].left, sorted_bullet_info_list[0][0].top), (player_rect.centerx, player_rect.centery))
        else:
            min_dist_with_bullet = float(1e30000)

        # trigger when bullet distance with player <= 100
        if min_dist_with_bullet <= 120:
            # pick the nearest bullet
            bullet_rect = sorted_bullet_info_list[0][0]
            bullet_direction = sorted_bullet_info_list[0][1]
            # distance with center x <= 20
            if abs(bullet_rect.centerx - player_rect.centerx) <= 25:
                # distance with center x <= 2
                if abs(bullet_rect.centerx - player_rect.centerx) <= 5:
                    # bullet direction to up, on player's bottom
                    if bullet_direction == 0 and bullet_rect.top > player_rect.top:
                        # add direction to down
                        directions.append(2)
                        # shoot
                        shoot = 1
                        #print('block bullet from down')
                    # direction to down, on player's top
                    if bullet_direction == 2 and bullet_rect.top < player_rect.top:
                        # add direction to up
                        directions.append(0)
                        # shoot
                        shoot = 1
                        #print('block bullet from up')
                # not too near
                else:
                    # if bullet on player's right
                    if bullet_rect.left > player_rect.centerx:
                        # go left
                        directions.append(3)
                        # go right
                        # directions.append(1)
                        #print('go left, skip bullet')
                    else:
                        # go right
                        directions.append(1)
                        # go left
                        # directions.append(3)
                        #print('go right, skip bullet')
            # distance with center y <= 20
            elif abs(bullet_rect.centery - player_rect.centery) <= 25:
                # distance with center y <= 2
                if abs(bullet_rect.centery - player_rect.centery) <= 5:
                    # bullet direction to right, on player's left
                    if bullet_direction == 1 and bullet_rect.left < player_rect.left:
                        # go left
                        directions.append(3)
                        # shoot
                        shoot = 1
                        #print('block bullet from left')
                    # bullet direction to left, on player's right
                    if bullet_direction == 3 and bullet_rect.left > player_rect.left:
                        # go right
                        directions.append(1)
                        # shoot
                        shoot = 1
                        #print('block bullet from right')
                # not too near
                else:
                    # on player bottom
                    if bullet_rect.top > player_rect.centery:
                        directions.append(0)
                        directions.append(2)
                        #print('go up, skip bullet')
                    else:
                        directions.append(2)
                        directions.append(0)
                        #print('go down, skip bullet')
            # neither distance with center x or center y <= 20
            else:
                # inline with enemy direction is same as a star direction
                if inlined_with_enemy == direction_from_astar:
                    shoot = 1
                directions.append(direction_from_astar)

                # bullet direction down or up
                if bullet_direction == 0 or bullet_direction == 2:
                    # bullet on right hand side
                    if bullet_rect.left > player_rect.left:
                        if 1 in directions:
                            directions.remove(1)
                        #print('bullet on rhs, don\'t go right')
                    else:
                        if 3 in directions:
                            directions.remove(3)
                        #print('bullet on lhs, don\'t go left')
                # bullet direction to left or right
                if bullet_direction == 1 or bullet_direction == 3:
                    # bullet on bottom
                    if bullet_rect.top > player_rect.top:
                        if 2 in directions:
                            directions.remove(2)
                        #print('bullet on bottom, don\'t go down')
                    else:
                        if 0 in directions:
                            directions.remove(0)
                        #print('bullt on top, don\'t go up')
        # distance with nearest bullet > 100 (threshold)
        else:
            # if inlined
            if inlined_with_enemy == direction_from_astar:
                shoot = 1
            directions.append(direction_from_astar)

        if directions:
            for direction in directions:
                # go up
                if direction == 0:
                    new_left = player_rect.left
                    new_top = player_rect.top - speed
                # go right
                elif direction == 1:
                    new_left = player_rect.left + speed
                    new_top = player_rect.top
                # go down
                elif direction == 2:
                    new_left = player_rect.left
                    new_top = player_rect.top + speed
                # go left
                elif direction == 3:
                    new_left = player_rect.left - speed
                    new_top = player_rect.top
                # no change
                else:
                    new_top = player_rect.top
                    new_left = player_rect.left

                temp_rect = pygame.Rect(new_left, new_top, 26, 26)
                # check collision with tile

                if 0 <= new_top <= 416 - 26 and 0 <= new_left <= 416 - 26:
                    collision = False
                    for tile_info in self.mapinfo[2]:
                        tile_rect = tile_info[0]
                        tile_type = tile_info[1]
                        if tile_type != 4:
                            if temp_rect.colliderect(tile_rect):
                                collision = True
                                break
                    if collision:
                        if inlined_with_enemy == direction_from_astar:
                            shoot = 1
                            break
                    else:
                        return shoot, direction
                    # collision = temp_rect.collidelist(obstacles)
                    # if collision:
                    #     if inlined_with_enemy == direction_from_astar:
                    #         shoot = 1
                    #         break
                    # else:
                    #     return shoot, direction
        # no direction appended
        else:
            return shoot, 4
        return shoot, direction_from_astar
# Import required libraries
import numpy as np
import os, pygame, time, random, uuid, sys
import matplotlib.pyplot as plt
import multiprocessing
import queue
from queue import Empty

# Import the additional training AI bot libraries
import heapq
import math

# Import gym required libraries
import gymnasium as gym
from gymnasium import spaces

from skimage.transform import rescale
from collections import deque
'''
===============================================================================================================================
															AI BOT CODE
===============================================================================================================================
'''

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
		global obs_flag_castle_danger, obs_flag_enemy_in_line, obs_distance_closest_enemy_to_castle, obs_distance_closest_enemy_to_player

		while True:
			obs_flag_castle_danger = 0
			obs_flag_enemy_in_line = 0
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
				obs_distance_closest_enemy_to_castle = self.manhattan_distance(sorted_enemy_with_distance_to_castle[0][0].topleft, self.castle_rect.topleft)
				obs_distance_closest_enemy_to_player = self.manhattan_distance(sorted_enemy_with_distance_to_player[0][0].topleft, player_rect.center)
				if obs_distance_closest_enemy_to_castle < 150:
					obs_flag_castle_danger = 1
					enemy_rect = sorted_enemy_with_distance_to_castle[0][0]
					enemy_direction = sorted_enemy_with_distance_to_castle[0][1]
				# else chase the nearest enemy to player
				else:
					obs_flag_castle_danger = 0
					enemy_rect = sorted_enemy_with_distance_to_player[0][0]
					enemy_direction = sorted_enemy_with_distance_to_player[0][1]

				# check if inline with enemy
				inline_direction = self.inline_with_enemy(player_rect, enemy_rect)
				if inline_direction is not False:
					obs_flag_enemy_in_line = 1

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
			except queue.empty:
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
		global obs_flag_bullet_avoidance_triggered
		obs_flag_bullet_avoidance_triggered = 0
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
			obs_flag_bullet_avoidance_triggered = 1
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

'''
===============================================================================================================================
													MAIN TANK BATTALION GAME
===============================================================================================================================
'''


# ITU-R 601-2 luma transform
def rgb_to_grayscale(rgb_array):
	# Define the weights for the RGB channels
	weights = np.array([0.2989, 0.5870, 0.1140])
	# Calculate the dot product of each pixel with the weights to get the grayscale value
	grayscale_array = np.dot(rgb_array[...,:3], weights)
	# Round the values and convert to uint8
	grayscale_array_noinfobar = grayscale_array[:, :416]
	grayscale_array_downscaled = rescale(grayscale_array_noinfobar, 1/2.0, anti_aliasing=True, mode='reflect')
	grayscale_array_rounded = np.round(grayscale_array_downscaled).astype(np.uint8)
	#grayscale_array_expanded = np.expand_dims(grayscale_array_rounded, axis=0)
	# print(grayscale_array)
	# plt.imshow(grayscale_array_rounded)
	# plt.show()
	# for i in range(grayscale_array.shape[0]):
	# 	for j in range(grayscale_array.shape[1]):
	# 			print(f"Value at ({i}, {j}): {grayscale_array[i, j]}")
	
	# print(grayscale_array.shape)
	# for i in range(grayscale_array.shape[0]):
	# 	for j in range(grayscale_array.shape[1]):
	# 		for k in range(grayscale_array.shape[2]):
	# 			print(f"Value at ({i}, {j}, {k}): {grayscale_array[i, j, k]}")
	# print(grayscale_array[0, :10, :10])

	return grayscale_array_rounded

# # Fuction for better training the model
# def assess_danger(bullets, player_rect, castle_rect):
#     # Constants representing directions
#     DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = range(4)

#     # Check each bullet
#     for bullet in bullets:
#         if bullet.owner == Bullet.OWNER_ENEMY:  # We only consider enemy bullets as danger
#             # Check if the bullet is moving towards the player's tank
#             if bullet.direction == DIR_DOWN and bullet.rect.top < player_rect.top and bullet.rect.left == player_rect.left:
#                 return 1
#             if bullet.direction == DIR_UP and bullet.rect.top > player_rect.top and bullet.rect.left == player_rect.left:
#                 return 1
#             if bullet.direction == DIR_RIGHT and bullet.rect.left < player_rect.left and bullet.rect.top == player_rect.top:
#                 return 1
#             if bullet.direction == DIR_LEFT and bullet.rect.left > player_rect.left and bullet.rect.top == player_rect.top:
#                 return 1

#             # Check if the bullet is moving towards the base
#             if bullet.direction == DIR_DOWN and bullet.rect.top < castle_rect.top and bullet.rect.left == castle_rect.left:
#                 return 1
#             if bullet.direction == DIR_UP and bullet.rect.top > castle_rect.top and bullet.rect.left == castle_rect.left:
#                 return 1
#             if bullet.direction == DIR_RIGHT and bullet.rect.left < castle_rect.left and bullet.rect.top == castle_rect.top:
#                 return 1
#             if bullet.direction == DIR_LEFT and bullet.rect.left > castle_rect.left and bullet.rect.top == castle_rect.top:
#                 return 1

#     # No bullets are a danger
#     return 0

# # Example usage:
# # Assuming bullets is a list of Bullet objects, player_rect is the player's rectangle, and castle_rect is the castle's rectangle
# #danger = assess_danger(bullets, player_rect, castle_rect)



class myRect(pygame.Rect):
	""" Add type property """
	def __init__(self, left, top, width, height, type):
		pygame.Rect.__init__(self, left, top, width, height)
		self.type = type

class Timer(object):
	def __init__(self):
		self.timers = []

	def add(self, interval, f, repeat = -1):
		options = {
			"interval"	: interval,
			"callback"	: f,
			"repeat"		: repeat,
			"times"			: 0,
			"time"			: 0,
			"uuid"			: uuid.uuid4()
		}
  
		self.timers.append(options)

		return options["uuid"]

	def destroy(self, uuid_nr):
		for timer in self.timers:
			if timer["uuid"] == uuid_nr:
				self.timers.remove(timer)
				return

	def update(self, time_passed):
		for timer in self.timers:
			timer["time"] += time_passed
			if timer["time"] > timer["interval"]:
				timer["time"] -= timer["interval"]
				timer["times"] += 1
				if timer["repeat"] > -1 and timer["times"] == timer["repeat"]:
					self.timers.remove(timer)
				try:
					timer["callback"]()
				except:
					try:
						self.timers.remove(timer)
					except:
						pass

class Castle():
	""" Player's castle/fortress """

	(STATE_STANDING, STATE_DESTROYED, STATE_EXPLODING) = range(3)

	def __init__(self):

		global sprites

		# images
		self.img_undamaged = sprites.subsurface(0, 15*2, 16*2, 16*2)
		self.img_destroyed = sprites.subsurface(16*2, 15*2, 16*2, 16*2)

		# init position
		self.rect = pygame.Rect(12*16, 24*16, 32, 32)

		# start w/ undamaged and shiny castle
		self.rebuild()

	def draw(self):
		""" Draw castle """
		global screen

		screen.blit(self.image, self.rect.topleft)

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DESTROYED
				del self.explosion
			else:
				self.explosion.draw()

	def rebuild(self):
		""" Reset castle """
		self.state = self.STATE_STANDING
		self.image = self.img_undamaged
		self.active = True

	def destroy(self):
		""" Destroy castle """
		self.state = self.STATE_EXPLODING
		self.explosion = Explosion(self.rect.topleft)
		self.image = self.img_destroyed
		self.active = False

class Bonus():
	""" Various power-ups
	When bonus is spawned, it begins flashing and after some time dissapears

	Available bonusses:
		grenade	: Picking up the grenade power up instantly wipes out ever enemy presently on the screen, including Armor Tanks regardless of how many times you've hit them. You do not, however, get credit for destroying them during the end-stage bonus points.
		helmet	: The helmet power up grants you a temporary force field that makes you invulnerable to enemy shots, just like the one you begin every stage with.
		shovel	: The shovel power up turns the walls around your fortress from brick to stone. This makes it impossible for the enemy to penetrate the wall and destroy your fortress, ending the game prematurely. The effect, however, is only temporary, and will wear off eventually.
		star		: The star power up grants your tank with new offensive power each time you pick one up, up to three times. The first star allows you to fire your bullets as fast as the power tanks can. The second star allows you to fire up to two bullets on the screen at one time. And the third star allows your bullets to destroy the otherwise unbreakable steel walls. You carry this power with you to each new stage until you lose a life.
		tank		: The tank power up grants you one extra life. The only other way to get an extra life is to score 20000 points.
		timer		: The timer power up temporarily freezes time, allowing you to harmlessly approach every tank and destroy them until the time freeze wears off.
	"""

	# bonus types
	(BONUS_GRENADE, BONUS_HELMET, BONUS_SHOVEL, BONUS_STAR, BONUS_TANK, BONUS_TIMER) = range(6)

	def __init__(self, level):

		global sprites

		# to know where to place
		self.level = level

		# bonus lives only for a limited period of time
		self.active = True

		# blinking state
		self.visible = True

		self.rect = pygame.Rect(random.randint(0, 416-32), random.randint(0, 416-32), 32, 32)

		self.bonus = random.choice([
			self.BONUS_GRENADE,
			self.BONUS_HELMET,
			self.BONUS_SHOVEL,
			self.BONUS_STAR,
			self.BONUS_TANK,
			self.BONUS_TIMER
		])

		self.image = sprites.subsurface(16*2*self.bonus, 32*2, 16*2, 15*2)

	def draw(self):
		""" draw bonus """
		global screen
		if self.visible:
			screen.blit(self.image, self.rect.topleft)

	def toggleVisibility(self):
		""" Toggle bonus visibility """
		self.visible = not self.visible


class Bullet():
	# direction constants
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	# bullet's stated
	(STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

	(OWNER_PLAYER, OWNER_ENEMY) = range(2)

	def __init__(self, level, position, direction, damage = 100, speed = 5):

		global sprites

		self.level = level
		self.direction = direction
		self.damage = damage
		self.owner = None
		self.owner_class = None

		# 1-regular everyday normal bullet
		# 2-can destroy steel
		self.power = 1

		self.image = sprites.subsurface(75*2, 74*2, 3*2, 4*2)

		# position is player's top left corner, so we'll need to
		# recalculate a bit. also rotate image itself.
		if direction == self.DIR_UP:
			self.rect = pygame.Rect(position[0] + 11, position[1] - 8, 6, 8)
		elif direction == self.DIR_RIGHT:
			self.image = pygame.transform.rotate(self.image, 270)
			self.rect = pygame.Rect(position[0] + 26, position[1] + 11, 8, 6)
		elif direction == self.DIR_DOWN:
			self.image = pygame.transform.rotate(self.image, 180)
			self.rect = pygame.Rect(position[0] + 11, position[1] + 26, 6, 8)
		elif direction == self.DIR_LEFT:
			self.image = pygame.transform.rotate(self.image, 90)
			self.rect = pygame.Rect(position[0] - 8 , position[1] + 11, 8, 6)

		self.explosion_images = [
			sprites.subsurface(0, 80*2, 32*2, 32*2),
			sprites.subsurface(32*2, 80*2, 32*2, 32*2),
		]

		self.speed = speed

		self.state = self.STATE_ACTIVE

	def draw(self):
		""" draw bullet """
		global screen
		if self.state == self.STATE_ACTIVE:
			screen.blit(self.image, self.rect.topleft)
		elif self.state == self.STATE_EXPLODING:
			self.explosion.draw()

	def update(self):
		global castle, players, enemies, bullets

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.destroy()
				del self.explosion

		if self.state != self.STATE_ACTIVE:
			return

		""" move bullet """
		if self.direction == self.DIR_UP:
			self.rect.topleft = [self.rect.left, self.rect.top - self.speed]
			if self.rect.top < 0:
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_RIGHT:
			self.rect.topleft = [self.rect.left + self.speed, self.rect.top]
			if self.rect.left > (416 - self.rect.width):
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_DOWN:
			self.rect.topleft = [self.rect.left, self.rect.top + self.speed]
			if self.rect.top > (416 - self.rect.height):
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_LEFT:
			self.rect.topleft = [self.rect.left - self.speed, self.rect.top]
			if self.rect.left < 0:
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return

		has_collided = False

		# check for collisions with walls. one bullet can destroy several (1 or 2)
		# tiles but explosion remains 1
		rects = self.level.obstacle_rects
		collisions = self.rect.collidelistall(rects)
		if collisions != []:
			for i in collisions:
				if self.level.hitTile(rects[i].topleft, self.power, self.owner == self.OWNER_PLAYER):
					has_collided = True
		if has_collided:
			self.explode()
			return

		# check for collisions with other bullets
		for bullet in bullets:
			if self.state == self.STATE_ACTIVE and bullet.owner != self.owner and bullet != self and self.rect.colliderect(bullet.rect):
				self.destroy()
				self.explode()
				return

		# check for collisions with players
		for player in players:
			if player.state == player.STATE_ALIVE and self.rect.colliderect(player.rect):
				if player.bulletImpact(self.owner == self.OWNER_PLAYER, self.damage, self.owner_class):
					self.destroy()
					return

		# check for collisions with enemies
		for enemy in enemies:
			if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
				if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
					self.destroy()
					return

		# check for collision with castle
		if castle.active and self.rect.colliderect(castle.rect):
			castle.destroy()
			self.destroy()
			return

	def explode(self):
		""" start bullets's explosion """
		global screen
		if self.state != self.STATE_REMOVED:
			self.state = self.STATE_EXPLODING
			self.explosion = Explosion([self.rect.left-13, self.rect.top-13], None, self.explosion_images)

	def destroy(self):
		self.state = self.STATE_REMOVED


class Label():
	def __init__(self, position, text = "", duration = None):

		self.position = position

		self.active = True

		self.text = text

		self.font = pygame.font.SysFont("Arial", 13)

		if duration != None:
			gtimer.add(duration, lambda :self.destroy(), 1)

	def draw(self):
		""" draw label """
		global screen
		screen.blit(self.font.render(self.text, False, (200,200,200)), [self.position[0]+4, self.position[1]+8])

	def destroy(self):
		self.active = False


class Explosion():
	def __init__(self, position, interval = None, images = None):

		global sprites

		self.position = [position[0]-16, position[1]-16]
		self.active = True

		if interval == None:
			interval = 100

		if images == None:
			images = [
				sprites.subsurface(0, 80*2, 32*2, 32*2),
				sprites.subsurface(32*2, 80*2, 32*2, 32*2),
				sprites.subsurface(64*2, 80*2, 32*2, 32*2)
			]

		images.reverse()

		self.images = [] + images

		self.image = self.images.pop()

		gtimer.add(interval, lambda :self.update(), len(self.images) + 1)

	def draw(self):
		global screen
		""" draw current explosion frame """
		screen.blit(self.image, self.position)

	def update(self):
		""" Advace to the next image """
		if len(self.images) > 0:
			self.image = self.images.pop()
		else:
			self.active = False

class Level():

	# tile constants
	(TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE) = range(6)

	# tile width/height in px
	TILE_SIZE = 16

	def __init__(self, level_nr = None):
		""" There are total 35 different levels. If level_nr is larger than 35, loop over
		to next according level so, for example, if level_nr ir 37, then load level 2 """

		global sprites

		# max number of enemies simultaneously  being on map
		self.max_active_enemies = 4

		tile_images = [
			pygame.Surface((8*2, 8*2)),
			sprites.subsurface(48*2, 64*2, 8*2, 8*2),
			sprites.subsurface(48*2, 72*2, 8*2, 8*2),
			sprites.subsurface(56*2, 72*2, 8*2, 8*2),
			sprites.subsurface(64*2, 64*2, 8*2, 8*2),
			sprites.subsurface(64*2, 64*2, 8*2, 8*2),
			sprites.subsurface(72*2, 64*2, 8*2, 8*2),
			sprites.subsurface(64*2, 72*2, 8*2, 8*2)
		]
		self.tile_empty = tile_images[0]
		self.tile_brick = tile_images[1]
		self.tile_steel = tile_images[2]
		self.tile_grass = tile_images[3]
		self.tile_water = tile_images[4]
		self.tile_water1= tile_images[4]
		self.tile_water2= tile_images[5]
		self.tile_froze = tile_images[6]

		self.obstacle_rects = []

		# level_nr = 1 if level_nr == None else level_nr%35
		# if level_nr == 0:
		# 	level_nr = 35

		self.loadLevel(level_nr)

		# tiles' rects on map, tanks cannot move over
		self.obstacle_rects = []

		# update these tiles
		self.updateObstacleRects()

		gtimer.add(400, lambda :self.toggleWaves())

	def hitTile(self, pos, power = 1, sound = False):
		"""
			Hit the tile
			@param pos Tile's x, y in px
			@return True if bullet was stopped, False otherwise
		"""

		global play_sounds, sounds

		for tile in self.mapr:
			if tile.topleft == pos:
				if tile.type == self.TILE_BRICK:
					if play_sounds and sound:
						sounds["brick"].play()
					self.mapr.remove(tile)
					self.updateObstacleRects()
					return True
				elif tile.type == self.TILE_STEEL:
					if play_sounds and sound:
						sounds["steel"].play()
					if power == 2:
						self.mapr.remove(tile)
						self.updateObstacleRects()
					return True
				else:
					return False

	def toggleWaves(self):
		""" Toggle water image """
		if self.tile_water == self.tile_water1:
			self.tile_water = self.tile_water2
		else:
			self.tile_water = self.tile_water1


	def loadLevel(self, level_nr = 1):
		""" Load specified level
		@return boolean Whether level was loaded
		"""
		filename = "levels/SeriousTry1/"+str(level_nr)
		if (not os.path.isfile(filename)):
			return False
		level = []
		f = open(filename, "r")
		data = f.read().split("\n")
		self.mapr = []
		x, y = 0, 0
		for row in data:
			for ch in row:
				if ch == "#":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_BRICK))
				elif ch == "@":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_STEEL))
				elif ch == "~":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_WATER))
				elif ch == "%":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_GRASS))
				elif ch == "-":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_FROZE))
				x += self.TILE_SIZE
			x = 0
			y += self.TILE_SIZE
		return True


	def draw(self, tiles = None):
		""" Draw specified map on top of existing surface """

		global screen

		if tiles == None:
			tiles = [TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE]

		for tile in self.mapr:
			if tile.type in tiles:
				if tile.type == self.TILE_BRICK:
					screen.blit(self.tile_brick, tile.topleft)
				elif tile.type == self.TILE_STEEL:
					screen.blit(self.tile_steel, tile.topleft)
				elif tile.type == self.TILE_WATER:
					screen.blit(self.tile_water, tile.topleft)
				elif tile.type == self.TILE_FROZE:
					screen.blit(self.tile_froze, tile.topleft)
				elif tile.type == self.TILE_GRASS:
					screen.blit(self.tile_grass, tile.topleft)

	def updateObstacleRects(self):
		""" Set self.obstacle_rects to all tiles' rects that players can destroy
		with bullets """

		global castle

		self.obstacle_rects = [castle.rect]

		for tile in self.mapr:
			if tile.type in (self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER):
				self.obstacle_rects.append(tile)

	def buildFortress(self, tile):
		""" Build walls around castle made from tile """

		positions = [
			(11*self.TILE_SIZE, 23*self.TILE_SIZE),
			(11*self.TILE_SIZE, 24*self.TILE_SIZE),
			(11*self.TILE_SIZE, 25*self.TILE_SIZE),
			(14*self.TILE_SIZE, 23*self.TILE_SIZE),
			(14*self.TILE_SIZE, 24*self.TILE_SIZE),
			(14*self.TILE_SIZE, 25*self.TILE_SIZE),
			(12*self.TILE_SIZE, 23*self.TILE_SIZE),
			(13*self.TILE_SIZE, 23*self.TILE_SIZE)
		]

		obsolete = []

		for i, rect in enumerate(self.mapr):
			if rect.topleft in positions:
				obsolete.append(rect)
		for rect in obsolete:
			self.mapr.remove(rect)

		for pos in positions:
			self.mapr.append(myRect(pos[0], pos[1], self.TILE_SIZE, self.TILE_SIZE, tile))

		self.updateObstacleRects()

class Tank():

	# possible directions
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	# states
	(STATE_SPAWNING, STATE_DEAD, STATE_ALIVE, STATE_EXPLODING) = range(4)

	# sides
	(SIDE_PLAYER, SIDE_ENEMY) = range(2)

	def __init__(self, level, side, position = None, direction = None, filename = None):

		global sprites

		# health. 0 health means dead
		self.health = 100

		# tank can't move but can rotate and shoot
		self.paralised = False

		# tank can't do anything
		self.paused = False

		# tank is protected from bullets
		self.shielded = False

		# px per move
		self.speed = 2

		# how many bullets can tank fire simultaneously
		self.max_active_bullets = 1

		# friend or foe
		self.side = side

		# flashing state. 0-off, 1-on
		self.flash = 0

		# 0 - no superpowers
		# 1 - faster bullets
		# 2 - can fire 2 bullets
		# 3 - can destroy steel
		self.superpowers = 0

		# each tank can pick up 1 bonus
		self.bonus = None

		# navigation keys: fire, up, right, down, left
		self.controls = [pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]

		# currently pressed buttons (navigation only)
		self.pressed = [False] * 4

		self.shield_images = [
			sprites.subsurface(0, 48*2, 16*2, 16*2),
			sprites.subsurface(16*2, 48*2, 16*2, 16*2)
		]
		self.shield_image = self.shield_images[0]
		self.shield_index = 0

		self.spawn_images = [
			sprites.subsurface(32*2, 48*2, 16*2, 16*2),
			sprites.subsurface(48*2, 48*2, 16*2, 16*2)
		]
		self.spawn_image = self.spawn_images[0]
		self.spawn_index = 0

		self.level = level

		if position != None:
			self.rect = pygame.Rect(position, (26, 26))
		else:
			self.rect = pygame.Rect(0, 0, 26, 26)

		if direction == None:
			self.direction = random.choice([self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT])
		else:
			self.direction = direction

		self.state = self.STATE_SPAWNING

		# spawning animation
		self.timer_uuid_spawn = gtimer.add(100, lambda :self.toggleSpawnImage())

		# duration of spawning
		self.timer_uuid_spawn_end = gtimer.add(1000, lambda :self.endSpawning())

	def endSpawning(self):
		""" End spawning
		Player becomes operational
		"""
		self.state = self.STATE_ALIVE
		gtimer.destroy(self.timer_uuid_spawn_end)


	def toggleSpawnImage(self):
		""" advance to the next spawn image """
		if self.state != self.STATE_SPAWNING:
			gtimer.destroy(self.timer_uuid_spawn)
			return
		self.spawn_index += 1
		if self.spawn_index >= len(self.spawn_images):
			self.spawn_index = 0
		self.spawn_image = self.spawn_images[self.spawn_index]

	def toggleShieldImage(self):
		""" advance to the next shield image """
		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_shield)
			return
		if self.shielded:
			self.shield_index += 1
			if self.shield_index >= len(self.shield_images):
				self.shield_index = 0
			self.shield_image = self.shield_images[self.shield_index]


	def draw(self):
		""" draw tank """
		global screen
		if self.state == self.STATE_ALIVE:
			screen.blit(self.image, self.rect.topleft)
			if self.shielded:
				screen.blit(self.shield_image, [self.rect.left-3, self.rect.top-3])
		elif self.state == self.STATE_EXPLODING:
			self.explosion.draw()
		elif self.state == self.STATE_SPAWNING:
			screen.blit(self.spawn_image, self.rect.topleft)

	def explode(self):
		""" start tanks's explosion """
		if self.state != self.STATE_DEAD:
			self.state = self.STATE_EXPLODING
			self.explosion = Explosion(self.rect.topleft)

			if self.bonus:
				self.spawnBonus()

	def fire(self, forced = False):
		""" Shoot a bullet
		@param boolean forced. If false, check whether tank has exceeded his bullet quota. Default: True
		@return boolean True if bullet was fired, false otherwise
		"""

		global bullets, labels

		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_fire)
			return False

		if self.paused:
			return False

		if not forced:
			active_bullets = 0
			for bullet in bullets:
				if bullet.owner_class == self and bullet.state == bullet.STATE_ACTIVE:
					active_bullets += 1
			if active_bullets >= self.max_active_bullets:
				return False

		bullet = Bullet(self.level, self.rect.topleft, self.direction)

		# if superpower level is at least 1
		if self.superpowers > 0:
			bullet.speed = 8

		# if superpower level is at least 3
		if self.superpowers > 2:
			bullet.power = 2

		if self.side == self.SIDE_PLAYER:
			bullet.owner = self.SIDE_PLAYER
		else:
			bullet.owner = self.SIDE_ENEMY
			self.bullet_queued = False

		bullet.owner_class = self
		bullets.append(bullet)
		return True

	def rotate(self, direction, fix_position = True):
		""" Rotate tank
		rotate, update image and correct position
		"""
		self.direction = direction

		if direction == self.DIR_UP:
			self.image = self.image_up
		elif direction == self.DIR_RIGHT:
			self.image = self.image_right
		elif direction == self.DIR_DOWN:
			self.image = self.image_down
		elif direction == self.DIR_LEFT:
			self.image = self.image_left

		if fix_position:
			new_x = self.nearest(self.rect.left, 8) + 3
			new_y = self.nearest(self.rect.top, 8) + 3

			if (abs(self.rect.left - new_x) < 5):
				self.rect.left = new_x

			if (abs(self.rect.top - new_y) < 5):
				self.rect.top = new_y

	def turnAround(self):
		""" Turn tank into opposite direction """
		if self.direction in (self.DIR_UP, self.DIR_RIGHT):
			self.rotate(self.direction + 2, False)
		else:
			self.rotate(self.direction - 2, False)

	def update(self, time_passed):
		""" Update timer and explosion (if any) """
		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DEAD
				del self.explosion

	def nearest(self, num, base):
		""" Round number to nearest divisible """
		return int(round(num / (base * 1.0)) * base)


	def bulletImpact(self, friendly_fire = False, damage = 100, tank = None):
		""" Bullet impact
		Return True if bullet should be destroyed on impact. Only enemy friendly-fire
		doesn't trigger bullet explosion
		"""

		global play_sounds, sounds

		if self.shielded:
			return True

		if not friendly_fire:
			self.health -= damage
			if self.health < 1:
				if self.side == self.SIDE_ENEMY:
					tank.trophies["enemy"+str(self.type)] += 1
					points = (self.type+1) * 100
					tank.score += points
					if play_sounds:
						sounds["explosion"].play()

					labels.append(Label(self.rect.topleft, str(points), 500))

				self.explode()
			return True

		if self.side == self.SIDE_ENEMY:
			return False
		elif self.side == self.SIDE_PLAYER:
			if not self.paralised:
				self.setParalised(True)
				self.timer_uuid_paralise = gtimer.add(10000, lambda :self.setParalised(False), 1)
			return True

	def setParalised(self, paralised = True):
		""" set tank paralise state
		@param boolean paralised
		@return None
		"""
		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_paralise)
			return
		self.paralised = paralised

class Enemy(Tank):

	(TYPE_BASIC, TYPE_FAST, TYPE_POWER, TYPE_ARMOR) = range(4)

	def __init__(self, level, type, position = None, direction = None, filename = None):

		Tank.__init__(self, level, type, position = None, direction = None, filename = None)

		global enemies, sprites

		# if true, do not fire
		self.bullet_queued = False

		# chose type on random
		if len(level.enemies_left) > 0:
			self.type = level.enemies_left.pop()
		else:
			self.state = self.STATE_DEAD
			return

		if self.type == self.TYPE_BASIC:
			self.speed = 1
		elif self.type == self.TYPE_FAST:
			self.speed = 3
		elif self.type == self.TYPE_POWER:
			self.superpowers = 1
		elif self.type == self.TYPE_ARMOR:
			self.health = 400

		# 1 in 5 chance this will be bonus carrier, but only if no other tank is
		if random.randint(1, 5) == 1:
			self.bonus = True
			for enemy in enemies:
				if enemy.bonus:
					self.bonus = False
					break

		images = [
			sprites.subsurface(32*2, 0, 13*2, 15*2),
			sprites.subsurface(48*2, 0, 13*2, 15*2),
			sprites.subsurface(64*2, 0, 13*2, 15*2),
			sprites.subsurface(80*2, 0, 13*2, 15*2),
			sprites.subsurface(32*2, 16*2, 13*2, 15*2),
			sprites.subsurface(48*2, 16*2, 13*2, 15*2),
			sprites.subsurface(64*2, 16*2, 13*2, 15*2),
			sprites.subsurface(80*2, 16*2, 13*2, 15*2)
		]

		self.image = images[self.type+0]

		self.image_up = self.image
		self.image_left = pygame.transform.rotate(self.image, 90)
		self.image_down = pygame.transform.rotate(self.image, 180)
		self.image_right = pygame.transform.rotate(self.image, 270)

		if self.bonus:
			self.image1_up = self.image_up
			self.image1_left = self.image_left
			self.image1_down = self.image_down
			self.image1_right = self.image_right

			self.image2 = images[self.type+4]
			self.image2_up = self.image2
			self.image2_left = pygame.transform.rotate(self.image2, 90)
			self.image2_down = pygame.transform.rotate(self.image2, 180)
			self.image2_right = pygame.transform.rotate(self.image2, 270)

		self.rotate(self.direction, False)

		if position == None:
			self.rect.topleft = self.getFreeSpawningPosition()
			if not self.rect.topleft:
				self.state = self.STATE_DEAD
				return

		# list of map coords where tank should go next
		self.path = self.generatePath(self.direction)

		# 1000 is duration between shots
		self.timer_uuid_fire = gtimer.add(1000, lambda :self.fire())

		# turn on flashing
		if self.bonus:
			self.timer_uuid_flash = gtimer.add(200, lambda :self.toggleFlash())

	def toggleFlash(self):
		""" Toggle flash state """
		if self.state not in (self.STATE_ALIVE, self.STATE_SPAWNING):
			gtimer.destroy(self.timer_uuid_flash)
			return
		self.flash = not self.flash
		if self.flash:
			self.image_up = self.image2_up
			self.image_right = self.image2_right
			self.image_down = self.image2_down
			self.image_left = self.image2_left
		else:
			self.image_up = self.image1_up
			self.image_right = self.image1_right
			self.image_down = self.image1_down
			self.image_left = self.image1_left
		self.rotate(self.direction, False)

	def spawnBonus(self):
		""" Create new bonus if needed """

		global bonuses

		if len(bonuses) > 0:
			return
		bonus = Bonus(self.level)
		bonuses.append(bonus)
		gtimer.add(500, lambda :bonus.toggleVisibility())
		gtimer.add(10000, lambda :bonuses.remove(bonus), 1)


	def getFreeSpawningPosition(self):

		global players, enemies

		available_positions = [
			[(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
			[12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
			[24 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,  (self.level.TILE_SIZE * 2 - self.rect.height) / 2]
		]

		random.shuffle(available_positions)

		for pos in available_positions:

			enemy_rect = pygame.Rect(pos, [26, 26])

			# collisions with other enemies
			collision = False
			for enemy in enemies:
				if enemy_rect.colliderect(enemy.rect):
					collision = True
					continue

			if collision:
				continue

			# collisions with players
			collision = False
			for player in players:
				if enemy_rect.colliderect(player.rect):
					collision = True
					continue

			if collision:
				continue

			return pos
		return False

	def move(self):
		""" move enemy if possible """

		global players, enemies, bonuses

		if self.state != self.STATE_ALIVE or self.paused or self.paralised:
			return

		if self.path == []:
			self.path = self.generatePath(None, True)

		new_position = self.path.pop(0)

		# move enemy
		if self.direction == self.DIR_UP:
			if new_position[1] < 0:
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_RIGHT:
			if new_position[0] > (416 - 26):
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_DOWN:
			if new_position[1] > (416 - 26):
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_LEFT:
			if new_position[0] < 0:
				self.path = self.generatePath(self.direction, True)
				return

		new_rect = pygame.Rect(new_position, [26, 26])

		# collisions with tiles
		if new_rect.collidelist(self.level.obstacle_rects) != -1:
			self.path = self.generatePath(self.direction, True)
			return

		# collisions with other enemies
		for enemy in enemies:
			if enemy != self and new_rect.colliderect(enemy.rect):
				self.turnAround()
				self.path = self.generatePath(self.direction)
				return

		# collisions with players
		for player in players:
			if new_rect.colliderect(player.rect):
				self.turnAround()
				self.path = self.generatePath(self.direction)
				return

		# collisions with bonuses
		for bonus in bonuses:
			if new_rect.colliderect(bonus.rect):
				bonuses.remove(bonus)

		# if no collision, move enemy
		self.rect.topleft = new_rect.topleft


	def update(self, time_passed):
		Tank.update(self, time_passed)
		if self.state == self.STATE_ALIVE and not self.paused:
			self.move()

	def generatePath(self, direction = None, fix_direction = False):
		""" If direction is specified, try continue that way, otherwise choose at random
		"""

		all_directions = [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT]

		if direction == None:
			if self.direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = self.direction + 2
			else:
				opposite_direction = self.direction - 2
			directions = all_directions
			random.shuffle(directions)
			directions.remove(opposite_direction)
			directions.append(opposite_direction)
		else:
			if direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = direction + 2
			else:
				opposite_direction = direction - 2

			if direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = direction + 2
			else:
				opposite_direction = direction - 2
			directions = all_directions
			random.shuffle(directions)
			directions.remove(opposite_direction)
			directions.remove(direction)
			directions.insert(0, direction)
			directions.append(opposite_direction)

		# at first, work with general units (steps) not px
		x = int(round(self.rect.left / 16))
		y = int(round(self.rect.top / 16))

		new_direction = None

		for direction in directions:
			if direction == self.DIR_UP and y > 1:
				new_pos_rect = self.rect.move(0, -8)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_RIGHT and x < 24:
				new_pos_rect = self.rect.move(8, 0)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_DOWN and y < 24:
				new_pos_rect = self.rect.move(0, 8)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_LEFT and x > 1:
				new_pos_rect = self.rect.move(-8, 0)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break

		# if we can go anywhere else, turn around
		if new_direction == None:
			new_direction = opposite_direction

		# fix tanks position
		if fix_direction and new_direction == self.direction:
			fix_direction = False

		self.rotate(new_direction, fix_direction)

		positions = []

		x = self.rect.left
		y = self.rect.top

		if new_direction in (self.DIR_RIGHT, self.DIR_LEFT):
			axis_fix = self.nearest(y, 16) - y
		else:
			axis_fix = self.nearest(x, 16) - x
		axis_fix = 0

		pixels = self.nearest(random.randint(1, 12) * 32, 32) + axis_fix + 3

		if new_direction == self.DIR_UP:
			for px in range(0, pixels, self.speed):
				positions.append([x, y-px])
		elif new_direction == self.DIR_RIGHT:
			for px in range(0, pixels, self.speed):
				positions.append([x+px, y])
		elif new_direction == self.DIR_DOWN:
			for px in range(0, pixels, self.speed):
				positions.append([x, y+px])
		elif new_direction == self.DIR_LEFT:
			for px in range(0, pixels, self.speed):
				positions.append([x-px, y])

		return positions

class Player(Tank):

	def __init__(self, level, type, position = None, direction = None, filename = None):

		Tank.__init__(self, level, type, position = None, direction = None, filename = None)

		global sprites

		if filename == None:
			filename = (0, 0, 16*2, 16*2)

		self.start_position = position
		self.start_direction = direction

		self.lives = 3 # Change the number of lives of the player

		# total score
		self.score = 0

		# store how many bonuses in this stage this player has collected
		self.trophies = {
			"bonus" : 0,
			"enemy0" : 0,
			"enemy1" : 0,
			"enemy2" : 0,
			"enemy3" : 0
		}

		self.image = sprites.subsurface(filename)
		self.image_up = self.image
		self.image_left = pygame.transform.rotate(self.image, 90)
		self.image_down = pygame.transform.rotate(self.image, 180)
		self.image_right = pygame.transform.rotate(self.image, 270)

		if direction == None:
			self.rotate(self.DIR_UP, False)
		else:
			self.rotate(direction, False)

	def move(self, direction):
		global obs_flag_player_collision
		""" move player if possible """

		global players, enemies, bonuses

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DEAD
				del self.explosion

		if self.state != self.STATE_ALIVE:
			return

		# rotate player
		if self.direction != direction:
			self.rotate(direction)

		if self.paralised:
			return

		# move player
		if direction == self.DIR_UP:
			new_position = [self.rect.left, self.rect.top - self.speed]
			if new_position[1] < 0:
				return
		elif direction == self.DIR_RIGHT:
			new_position = [self.rect.left + self.speed, self.rect.top]
			if new_position[0] > (416 - 26):
				return
		elif direction == self.DIR_DOWN:
			new_position = [self.rect.left, self.rect.top + self.speed]
			if new_position[1] > (416 - 26):
				return
		elif direction == self.DIR_LEFT:
			new_position = [self.rect.left - self.speed, self.rect.top]
			if new_position[0] < 0:
				return

		player_rect = pygame.Rect(new_position, [26, 26])

		# collisions with tiles
		if player_rect.collidelist(self.level.obstacle_rects) != -1:
			obs_flag_player_collision = 1
			return

		# collisions with other players
		for player in players:
			obs_flag_player_collision = 1
			if player != self and player.state == player.STATE_ALIVE and player_rect.colliderect(player.rect) == True:
				return

		# collisions with enemies
		for enemy in enemies:
			obs_flag_player_collision = 1
			if player_rect.colliderect(enemy.rect) == True:
				return

		# collisions with bonuses
		for bonus in bonuses:
			if player_rect.colliderect(bonus.rect) == True:
				self.bonus = bonus

		#if no collision, move player
		self.rect.topleft = (new_position[0], new_position[1])

	def reset(self, pos):
		""" reset player """
		self.start_position = pos
		self.start_direction = random.randint(0, 3)
		self.rotate(self.start_direction, False)
		self.rect.topleft = self.start_position
		self.superpowers = 0
		self.max_active_bullets = 1
		self.health = 100
		self.paralised = False
		self.paused = False
		self.pressed = [False] * 4
		self.state = self.STATE_ALIVE

		# """ reset player """
		# Create a list of valid x spawns if y > 21
		# valid_x_spawns = list(range(0, 10)) + list(range(15, 25))
		# TILE_SIZE = 16
  
		# # # Initialize first player randomly each level
		# # random_y = random.randint(0, 24)	
		# # if random_y > 21:
		# # 	random_x = random.choice(valid_x_spawns)
		# # else:
		# # 	random_x = random.randint(0, 24)
		# random_y = random.choice([0, 24])
		# if random_y == 24:
		# 	random_x = random.choice([8, 16])
		# 	x = random_x * TILE_SIZE + (TILE_SIZE * 2 - 26) / 2 # 8  #BLABLA
		# 	y = random_y * TILE_SIZE + (TILE_SIZE * 2 - 26) / 2 # 24
		# if random_y == 0:
		# 	random_x = random.choice([0.5, 24.5])
		# 	x = random_x * TILE_SIZE + (TILE_SIZE * 2 - 26) / 2 # 8  #BLABLA
		# 	y = random_y * TILE_SIZE + (TILE_SIZE * 2 - 26) / 2 # 24
		# self.start_position = [x, y]
		# self.rotate(self.start_direction, False)
		# self.rect.topleft = self.start_position
		# self.superpowers = 0
		# self.max_active_bullets = 1
		# self.health = 100
		# self.paralised = False
		# self.paused = False
		# self.pressed = [False] * 4
		# self.state = self.STATE_ALIVE

class Game():
	# direction constants
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	TILE_SIZE = 16

	def __init__(self):

		global screen, sprites, play_sounds, sounds

		# center window
		os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'

		if play_sounds:
			pygame.mixer.pre_init(44100, -16, 1, 512)

		pygame.init()


		pygame.display.set_caption("Battle City")

		size = width, height = 480, 416

		if "-f" in sys.argv[1:]:
			screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
		else:
			# for looking at game
			screen = pygame.display.set_mode(size)
			# for training
			#screen = pygame.display.set_mode(size, pygame.HIDDEN)

		self.clock = pygame.time.Clock()

		# load sprites (funky version)
		#sprites = pygame.transform.scale2x(pygame.image.load("images/sprites.gif"))
		# load sprites (pixely version)
		sprites = pygame.transform.scale(pygame.image.load("images/sprites.gif"), [192, 224])
		#screen.set_colorkey((0,138,104))

		pygame.display.set_icon(sprites.subsurface(0, 0, 13*2, 13*2))

		# load sounds
		if play_sounds:
			pygame.mixer.init(44100, -16, 1, 512)

			sounds["start"] = pygame.mixer.Sound("sounds/gamestart.ogg")
			sounds["end"] = pygame.mixer.Sound("sounds/gameover.ogg")
			sounds["score"] = pygame.mixer.Sound("sounds/score.ogg")
			sounds["bg"] = pygame.mixer.Sound("sounds/background.ogg")
			sounds["fire"] = pygame.mixer.Sound("sounds/fire.ogg")
			sounds["bonus"] = pygame.mixer.Sound("sounds/bonus.ogg")
			sounds["explosion"] = pygame.mixer.Sound("sounds/explosion.ogg")
			sounds["brick"] = pygame.mixer.Sound("sounds/brick.ogg")
			sounds["steel"] = pygame.mixer.Sound("sounds/steel.ogg")

		self.enemy_life_image = sprites.subsurface(81*2, 57*2, 7*2, 7*2)
		self.player_life_image = sprites.subsurface(89*2, 56*2, 7*2, 8*2)
		self.flag_image = sprites.subsurface(64*2, 49*2, 16*2, 15*2)

		# this is used in intro screen
		self.player_image = pygame.transform.rotate(sprites.subsurface(0, 0, 13*2, 13*2), 270)

		# if true, no new enemies will be spawn during this time
		self.timefreeze = False

		# load custom font
		self.font = pygame.font.Font("fonts/prstart.ttf", 16)

		# pre-render game over text
		self.im_game_over = pygame.Surface((64, 40))
		self.im_game_over.set_colorkey((0,0,0))
		self.im_game_over.blit(self.font.render("GAME", False, (127, 64, 64)), [0, 0])
		self.im_game_over.blit(self.font.render("OVER", False, (127, 64, 64)), [0, 20])
		self.game_over_y = 416+40

		# number of players. here is defined preselected menu value
		self.nr_of_players = 1
		self.available_positions = []

		del players[:]
		del bullets[:]
		del enemies[:]
		del bonuses[:]


	def triggerBonus(self, bonus, player):
		""" Execute bonus powers """

		global enemies, labels, play_sounds, sounds

		if play_sounds:
			sounds["bonus"].play()

		player.trophies["bonus"] += 1
		player.score += 500

		if bonus.bonus == bonus.BONUS_GRENADE:
			for enemy in enemies:
				enemy.explode()
		elif bonus.bonus == bonus.BONUS_HELMET:
			self.shieldPlayer(player, True, 10000)
		elif bonus.bonus == bonus.BONUS_SHOVEL:
			self.level.buildFortress(self.level.TILE_STEEL)
			gtimer.add(10000, lambda :self.level.buildFortress(self.level.TILE_BRICK), 1)
		elif bonus.bonus == bonus.BONUS_STAR:
			player.superpowers += 1
			if player.superpowers == 2:
				player.max_active_bullets = 2
		elif bonus.bonus == bonus.BONUS_TANK:
			player.lives += 1
		elif bonus.bonus == bonus.BONUS_TIMER:
			self.toggleEnemyFreeze(True)
			gtimer.add(10000, lambda :self.toggleEnemyFreeze(False), 1)
		bonuses.remove(bonus)

		labels.append(Label(bonus.rect.topleft, "500", 500))

	def shieldPlayer(self, player, shield = True, duration = None):
		""" Add/remove shield
		player: player (not enemy)
		shield: true/false
		duration: in ms. if none, do not remove shield automatically
		"""
		player.shielded = shield
		if shield:
			player.timer_uuid_shield = gtimer.add(100, lambda :player.toggleShieldImage())
		else:
			gtimer.destroy(player.timer_uuid_shield)

		if shield and duration != None:
			gtimer.add(duration, lambda :self.shieldPlayer(player, False), 1)


	def spawnEnemy(self):
		""" Spawn new enemy if needed
		Only add enemy if:
			- there are at least one in queue
			- map capacity hasn't exceeded its quota
			- now isn't timefreeze
		"""

		global enemies

		if len(enemies) >= self.level.max_active_enemies:
			return
		if len(self.level.enemies_left) < 1 or self.timefreeze:
			return
		enemy = Enemy(self.level, 1)

		enemies.append(enemy)


	def respawnPlayer(self, player, clear_scores = False):
		""" Respawn player """
		n = random.randint(0, len(self.available_positions) - 1)
		[kx, ky] = self.available_positions[n]
		x = kx * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
		y = ky * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
		pos = [x, y]

		player.reset(pos)

		if clear_scores:
			player.trophies = {
				"bonus" : 0, "enemy0" : 0, "enemy1" : 0, "enemy2" : 0, "enemy3" : 0
			}

		self.shieldPlayer(player, True, 4000)

	def gameOver(self):
		""" End game and return to menu """
		
		global play_sounds, sounds

		for player in players:
			player.lives = 3 # Change the number of lives

		
		if play_sounds:
			for sound in sounds:
				sounds[sound].stop()
			sounds["end"].play()

		self.game_over_y = 416+40

		self.game_over = True
		#gtimer.add(3000, lambda :self.showScores(), 1)
		if self.game_over:
			self.stage = 0
			self.nextLevel()
			# self.gameOverScreen() #VLADYS DONE FLAG
		else:
			self.nextLevel()

	def gameOverScreen(self):
		""" Show game over screen """

		global screen

		# stop game main loop (if any)
		self.running = False

		screen.fill([0, 0, 0])

		self.writeInBricks("game", [125, 140])
		self.writeInBricks("over", [125, 220])
		pygame.display.flip()

		while 1:
			time_passed = self.clock.tick(50)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					quit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RETURN:
						self.showMenu()
						return

	def showMenu(self):
		""" Show game menu
		Redraw screen only when up or down key is pressed. When enter is pressed,
		exit from this screen and start the game with selected number of players
		"""

		global players, screen, gtimer

		# stop game main loop (if any)
		self.running = False

		# clear all timers
		del gtimer.timers[:]

		# set current stage to 0
		self.stage = 0

		# ELIMINATED INTRO MENU
  
		# self.animateIntroScreen()
		# main_loop = True
		# while main_loop:
		# 	time_passed = self.clock.tick(50)

		# 	for event in pygame.event.get():
		# 		if event.type == pygame.QUIT:
		# 			quit()
		# 		elif event.type == pygame.KEYDOWN:
		# 			if event.key == pygame.K_q:
		# 				quit()
		# 			elif event.key == pygame.K_UP:
		# 				if self.nr_of_players == 2:
		# 					self.nr_of_players = 1
		# 					self.drawIntroScreen()
		# 			elif event.key == pygame.K_DOWN:
		# 				if self.nr_of_players == 1:
		# 					self.nr_of_players = 2
		# 					self.drawIntroScreen()
		# 			elif event.key == pygame.K_RETURN:
		# 				main_loop = False
		self.nr_of_players = 1
		del players[:]
		self.nextLevel()
		

	def reloadPlayers(self):
		""" Init players
		If players already exist, just reset them
		"""

		global players
  
		# Create a list of valid x spawns if y > 21
		#valid_x_spawns = list(range(0, 10)) + list(range(15, 25))

		if len(players) == 0:
			# choice = random.randint(1, 4)
			# if choice == 1:
				# first player
			x = 8 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2

			player = Player(
				self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			)
			players.append(player)
			# # Initialize first player randomly each level
			# random_y = random.randint(0, 24)
			# if random_y > 21:
			# 	random_x = random.choice(valid_x_spawns)
			# else:
			# 	random_x = random.randint(0, 24)
			
			# x = random_x * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2 # 8
			# y = random_y * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2 # 24

			# player = Player(
			# 	self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			# )
			# players.append(player)

			# second player
			# if choice == 2:
			# 	x = 16 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	player = Player(
			# 		self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			# 	)
			# 	player.controls = [102, 119, 100, 115, 97]
			# 	players.append(player)

			# if choice == 3:
			# 	x = 0 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	y = 0 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	player = Player(
			# 		self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			# 	)
			# 	player.controls = [102, 119, 100, 115, 97]
			# 	players.append(player)

			# if choice == 4:
			# 	x = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	y = 0 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			# 	player = Player(
			# 		self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			# 	)
			# 	player.controls = [102, 119, 100, 115, 97]
			# 	players.append(player)

		for player in players:
			player.level = self.level
			self.respawnPlayer(player, True)

	def showScores(self):
		""" Show level scores """

		global screen, sprites, players, play_sounds, sounds

		# stop game main loop (if any)
		self.running = False

		# clear all timers
		del gtimer.timers[:]

		if play_sounds:
			for sound in sounds:
				sounds[sound].stop()

		hiscore = self.loadHiscore()

		# update hiscore if needed
		if players[0].score > hiscore:
			hiscore = players[0].score
			self.saveHiscore(hiscore)
		if self.nr_of_players == 2 and players[1].score > hiscore:
			hiscore = players[1].score
			self.saveHiscore(hiscore)

		img_tanks = [
			sprites.subsurface(32*2, 0, 13*2, 15*2),
			sprites.subsurface(48*2, 0, 13*2, 15*2),
			sprites.subsurface(64*2, 0, 13*2, 15*2),
			sprites.subsurface(80*2, 0, 13*2, 15*2)
		]

		img_arrows = [
			sprites.subsurface(81*2, 48*2, 7*2, 7*2),
			sprites.subsurface(88*2, 48*2, 7*2, 7*2)
		]

		screen.fill([0, 0, 0])

		# colors
		black = pygame.Color("black")
		white = pygame.Color("white")
		purple = pygame.Color(127, 64, 64)
		pink = pygame.Color(191, 160, 128)

		screen.blit(self.font.render("HI-SCORE", False, purple), [105, 35])
		screen.blit(self.font.render(str(hiscore), False, pink), [295, 35])

		screen.blit(self.font.render("STAGE"+str(self.stage).rjust(3), False, white), [170, 65])

		screen.blit(self.font.render("I-PLAYER", False, purple), [25, 95])

		#player 1 global score
		screen.blit(self.font.render(str(players[0].score).rjust(8), False, pink), [25, 125])

		if self.nr_of_players == 2:
			screen.blit(self.font.render("II-PLAYER", False, purple), [310, 95])

			#player 2 global score
			screen.blit(self.font.render(str(players[1].score).rjust(8), False, pink), [325, 125])

		# tanks and arrows
		for i in range(4):
			screen.blit(img_tanks[i], [226, 160+(i*45)])
			screen.blit(img_arrows[0], [206, 168+(i*45)])
			if self.nr_of_players == 2:
				screen.blit(img_arrows[1], [258, 168+(i*45)])

		screen.blit(self.font.render("TOTAL", False, white), [70, 335])

		# total underline
		pygame.draw.line(screen, white, [170, 330], [307, 330], 4)

		pygame.display.flip()

		self.clock.tick(2)

		interval = 5

		# points and kills
		for i in range(4):

			# total specific tanks
			tanks = players[0].trophies["enemy"+str(i)]

			for n in range(tanks+1):
				if n > 0 and play_sounds:
					sounds["score"].play()

				# erase previous text
				screen.blit(self.font.render(str(n-1).rjust(2), False, black), [170, 168+(i*45)])
				# print new number of enemies
				screen.blit(self.font.render(str(n).rjust(2), False, white), [170, 168+(i*45)])
				# erase previous text
				screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [25, 168+(i*45)])
				# print new total points per enemy
				screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [25, 168+(i*45)])
				pygame.display.flip()
				self.clock.tick(interval)

			if self.nr_of_players == 2:
				tanks = players[1].trophies["enemy"+str(i)]

				for n in range(tanks+1):

					if n > 0 and play_sounds:
						sounds["score"].play()

					screen.blit(self.font.render(str(n-1).rjust(2), False, black), [277, 168+(i*45)])
					screen.blit(self.font.render(str(n).rjust(2), False, white), [277, 168+(i*45)])

					screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [325, 168+(i*45)])
					screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [325, 168+(i*45)])

					pygame.display.flip()
					self.clock.tick(interval)

			self.clock.tick(interval)

		# total tanks
		tanks = sum([i for i in players[0].trophies.values()]) - players[0].trophies["bonus"]
		screen.blit(self.font.render(str(tanks).rjust(2), False, white), [170, 335])
		if self.nr_of_players == 2:
			tanks = sum([i for i in players[1].trophies.values()]) - players[1].trophies["bonus"]
			screen.blit(self.font.render(str(tanks).rjust(2), False, white), [277, 335])

		pygame.display.flip()

		# do nothing for 2 seconds
		self.clock.tick(1)
		self.clock.tick(1)

		if self.game_over:
			self.gameOverScreen()
		else:
			self.nextLevel()


	def draw(self):
		global screen, castle, players, enemies, bullets, bonuses

		screen.fill([0, 0, 0])

		self.level.draw([self.level.TILE_EMPTY, self.level.TILE_BRICK, self.level.TILE_STEEL, self.level.TILE_FROZE, self.level.TILE_WATER])

		castle.draw()

		for enemy in enemies:
			enemy.draw()

		for label in labels:
			label.draw()

		for player in players:
			player.draw()

		for bullet in bullets:
			bullet.draw()

		for bonus in bonuses:
			bonus.draw()

		self.level.draw([self.level.TILE_GRASS])

		if self.game_over:
			if self.game_over_y > 188:
				self.game_over_y -= 4
			screen.blit(self.im_game_over, [176, self.game_over_y]) # 176=(416-64)/2

		self.drawSidebar() #VLADYS CHECK

		pygame.display.flip()

	def drawSidebar(self):

		global screen, players, enemies

		x = 416
		y = 0
		screen.fill([100, 100, 100], pygame.Rect([416, 0], [64, 416]))

		xpos = x + 16
		ypos = y + 16

		# draw enemy lives
		for n in range(len(self.level.enemies_left) + len(enemies)):
			screen.blit(self.enemy_life_image, [xpos, ypos])
			if n % 2 == 1:
				xpos = x + 16
				ypos+= 17
			else:
				xpos += 17

		# players' lives
		if pygame.font.get_init():
			text_color = pygame.Color('black')
			for n in range(len(players)):
				if n == 0:
					screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+200])
					screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+215])
					screen.blit(self.player_life_image, [x+17, y+215])
				else:
					screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+240])
					screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+255])
					screen.blit(self.player_life_image, [x+17, y+255])

			screen.blit(self.flag_image, [x+17, y+280])
			screen.blit(self.font.render(str(self.stage), False, text_color), [x+17, y+312])


	def drawIntroScreen(self, put_on_surface = True):
		""" Draw intro (menu) screen
		@param boolean put_on_surface If True, flip display after drawing
		@return None
		"""

		global screen

		screen.fill([0, 0, 0])

		if pygame.font.get_init():

			hiscore = self.loadHiscore()

			screen.blit(self.font.render("HI- "+str(hiscore), True, pygame.Color('white')), [170, 35])

			screen.blit(self.font.render("1 PLAYER", True, pygame.Color('white')), [165, 250])
			screen.blit(self.font.render("2 PLAYERS", True, pygame.Color('white')), [165, 275])

			screen.blit(self.font.render("(c) 1980 1985 NAMCO LTD.", True, pygame.Color('white')), [50, 350])
			screen.blit(self.font.render("ALL RIGHTS RESERVED", True, pygame.Color('white')), [85, 380])


		if self.nr_of_players == 1:
			screen.blit(self.player_image, [125, 245])
		elif self.nr_of_players == 2:
			screen.blit(self.player_image, [125, 270])

		self.writeInBricks("battle", [65, 80])
		self.writeInBricks("city", [129, 160])

		if put_on_surface:
			pygame.display.flip()

	def animateIntroScreen(self):
		""" Slide intro (menu) screen from bottom to top
		If Enter key is pressed, finish animation immediately
		@return None
		"""

		global screen

		self.drawIntroScreen(False)
		screen_cp = screen.copy()

		screen.fill([0, 0, 0])

		y = 416
		while (y > 0):
			time_passed = self.clock.tick(50)
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RETURN:
						y = 0
						break

			screen.blit(screen_cp, [0, y])
			pygame.display.flip()
			y -= 5

		screen.blit(screen_cp, [0, 0])
		pygame.display.flip()


	def chunks(self, l, n):
		""" Split text string in chunks of specified size
		@param string l Input string
		@param int n Size (number of characters) of each chunk
		@return list
		"""
		return [l[i:i+n] for i in range(0, len(l), n)]

	def writeInBricks(self, text, pos):
		""" Write specified text in "brick font"
		Only those letters are available that form words "Battle City" and "Game Over"
		Both lowercase and uppercase are valid input, but output is always uppercase
		Each letter consists of 7x7 bricks which is converted into 49 character long string
		of 1's and 0's which in turn is then converted into hex to save some bytes
		@return None
		"""

		global screen, sprites

		bricks = sprites.subsurface(56*2, 64*2, 8*2, 8*2)
		brick1 = bricks.subsurface((0, 0, 8, 8))
		brick2 = bricks.subsurface((8, 0, 8, 8))
		brick3 = bricks.subsurface((8, 8, 8, 8))
		brick4 = bricks.subsurface((0, 8, 8, 8))

		alphabet = {
			"a" : "0071b63c7ff1e3",
			"b" : "01fb1e3fd8f1fe",
			"c" : "00799e0c18199e",
			"e" : "01fb060f98307e",
			"g" : "007d860cf8d99f",
			"i" : "01f8c183060c7e",
			"l" : "0183060c18307e",
			"m" : "018fbffffaf1e3",
			"o" : "00fb1e3c78f1be",
			"r" : "01fb1e3cff3767",
			"t" : "01f8c183060c18",
			"v" : "018f1e3eef8e08",
			"y" : "019b3667860c18"
		}

		abs_x, abs_y = pos

		for letter in text.lower():

			binstr = ""
			for h in self.chunks(alphabet[letter], 2):
				binstr += str(bin(int(h, 16)))[2:].rjust(8, "0")
			binstr = binstr[7:]

			x, y = 0, 0
			letter_w = 0
			surf_letter = pygame.Surface((56, 56))
			for j, row in enumerate(self.chunks(binstr, 7)):
				for i, bit in enumerate(row):
					if bit == "1":
						if i%2 == 0 and j%2 == 0:
							surf_letter.blit(brick1, [x, y])
						elif i%2 == 1 and j%2 == 0:
							surf_letter.blit(brick2, [x, y])
						elif i%2 == 1 and j%2 == 1:
							surf_letter.blit(brick3, [x, y])
						elif i%2 == 0 and j%2 == 1:
							surf_letter.blit(brick4, [x, y])
						if x > letter_w:
							letter_w = x
					x += 8
				x = 0
				y += 8
			screen.blit(surf_letter, [abs_x, abs_y])
			abs_x += letter_w + 16

	def toggleEnemyFreeze(self, freeze = True):
		""" Freeze/defreeze all enemies """

		global enemies

		for enemy in enemies:
			enemy.paused = freeze
		self.timefreeze = freeze


	def loadHiscore(self):
		""" Load hiscore
		Really primitive version =] If for some reason hiscore cannot be loaded, return 20000
		@return int
		"""
		filename = ".hiscore"
		if (not os.path.isfile(filename)):
			return 20000

		f = open(filename, "r")
		hiscore = int(f.read())

		if hiscore > 19999 and hiscore < 1000000:
			return hiscore
		else:
			
			return 20000

	def saveHiscore(self, hiscore):
		""" Save hiscore
		@return boolean
		"""
		try:
			f = open(".hiscore", "w")
		except:
			
			return False
		f.write(str(hiscore))
		f.close()
		return True


	def finishLevel(self):
		""" Finish current level
		Show earned scores and advance to the next stage
		"""

		global play_sounds, sounds

		for player in players:
			player.lives = 3 # Chage the number of lives of the player
			
		if play_sounds:
			sounds["bg"].stop()

		self.active = False
		#gtimer.add(3000, lambda :self.showScores(), 1)
		if self.game_over:
			game.showMenu()
		else:
			self.nextLevel()		
		print("Stage "+str(self.stage)+" completed")

	def nextLevel(self):
		""" Start next level """

		global castle, players, bullets, bonuses, play_sounds, sounds, screen_array, screen_array_grayscale

		del bullets[:]
		del enemies[:]
		del bonuses[:]
		castle.rebuild()
		del gtimer.timers[:]

		# Load a random level
		self.stage = random.randint(1, 1000)
		self.level = Level(self.stage)
		self.timefreeze = False

		# Generate a random number of type 1 tanks for the level
		num_enemies = random.randint(6, 20)
		self.level.enemies_left = [0] * num_enemies  # List with 'num_enemies' instances of type 1 tanks

		if play_sounds:
			sounds["start"].play()
			gtimer.add(4330, lambda :sounds["bg"].play(-1), 1)

		###################### Code for random initialization - DANI ######################
		self.available_positions = []
		filename = "levels/SeriousTry1/" + str(self.stage)
		f = open(filename, "r")
		data = f.read().split("\n")
		f.close()
		for y in range(len(data) - 1):
			row = data[y]
			for x in range(len(row) - 1):
				if row[x] == "." and row[x + 1] == "." and data[y + 1][x] == "." and data[y+1][x+1] == "." and not (x == 12 and y == 24):
					self.available_positions.append([x, y])
		random.shuffle(self.available_positions)	
		###################### Code for random initialization - DANI ######################

		self.reloadPlayers()

		gtimer.add(2500, lambda :self.spawnEnemy()) #CHECK

		# if True, start "game over" animation
		self.game_over = False #VLADYS DONE FLAG

		# if False, game will end w/o "game over" bussiness
		self.running = True

		# if False, players won't be able to do anything
		self.active = True

		self.draw() #VLADYS RENDER 

		screen_array = pygame.surfarray.array3d(screen)
		screen_array = np.transpose(screen_array, (1, 0, 2))
		screen_array_grayscale = rgb_to_grayscale(screen_array)

		# Initialize AI training bot
		self.agent = ai_agent()
		self.p_mapinfo = multiprocessing.Queue()
		self.c_control = multiprocessing.Queue()
  
		mapinfo = self.get_mapinfo()
		self.agent.mapinfo = mapinfo
		if self.p_mapinfo.empty() == True:
			self.p_mapinfo.put(mapinfo)

		self.ai_bot_actions = [0, 4]
		self.p = multiprocessing.Process(target = self.agent.operations, args = (self.p_mapinfo, self.c_control))
		self.p.start()
  
		# Display the array using matplotlib
		# plt.imshow(screen_array_grayscale)
		# plt.show()
		# while self.running:
		# 	if step_flag == True:
		# 		step_flag = False
		# 		time_passed = self.clock.tick(50)
		# 		for player in players:
		# 			if player.state == player.STATE_ALIVE and not self.game_over and self.active:
		# 				if action_global == 1: #action_global == 0 is doing nothing
		# 					player.fire()
		# 				elif action_global == 2:
		# 					player.move(self.DIR_UP);
		# 				elif action_global == 3:
		# 					player.move(self.DIR_RIGHT);
		# 				elif action_global == 4:
		# 					player.move(self.DIR_DOWN);
		# 				elif action_global == 5:
		# 					player.move(self.DIR_LEFT);
		# 			player.update(time_passed)

		# 		for enemy in enemies:
		# 			if enemy.state == enemy.STATE_DEAD and not self.game_over and self.active:
		# 				enemies.remove(enemy)
		# 				if len(self.level.enemies_left) == 0 and len(enemies) == 0:
		# 					self.finishLevel()
		# 			else:
		# 				enemy.update(time_passed)

		# 		if not self.game_over and self.active:
		# 			for player in players:
		# 				if player.state == player.STATE_ALIVE:
		# 					if player.bonus != None and player.side == player.SIDE_PLAYER:
		# 						self.triggerBonus(bonus, player)
		# 						player.bonus = None
		# 				elif player.state == player.STATE_DEAD:
		# 					self.superpowers = 0
		# 					player.lives -= 1
		# 					if player.lives > 0:
		# 						self.respawnPlayer(player)
		# 					else:
		# 						self.gameOver()

		# 		for bullet in bullets:
		# 			if bullet.state == bullet.STATE_REMOVED:
		# 				bullets.remove(bullet)
		# 			else:
		# 				bullet.update()

		# 		for bonus in bonuses:
		# 			if bonus.active == False:
		# 				bonuses.remove(bonus)

		# 		for label in labels:
		# 			if not label.active:
		# 				labels.remove(label)

		# 		if not self.game_over:
		# 			if not castle.active:
		# 				self.gameOver()

		# 		gtimer.update(time_passed)

		# 		self.draw() #VLADYS RENDER
		# 		screen_array = pygame.surfarray.array3d(screen)
		# 		screen_array = np.transpose(screen_array, (1, 0, 2))
		# 		screen_array_grayscale = rgb_to_grayscale(screen_array)

				# Display the array using matplotlib
				# plt.imshow(screen_array_grayscale)
				# plt.show()

		# while self.running:

		# 	time_passed = self.clock.tick(50)

		# 	for event in pygame.event.get():
		# 		if event.type == pygame.MOUSEBUTTONDOWN:
		# 			pass
		# 		elif event.type == pygame.QUIT:
		# 			quit()
		# 		elif event.type == pygame.KEYDOWN and not self.game_over and self.active:

		# 			if event.key == pygame.K_q:
		# 				quit()
		# 			# toggle sounds
		# 			elif event.key == pygame.K_m:
		# 				play_sounds = not play_sounds
		# 				if not play_sounds:
		# 					pygame.mixer.stop()
		# 				else:
		# 					sounds["bg"].play(-1)

		# 			for player in players:
		# 				if player.state == player.STATE_ALIVE:
		# 					try:
		# 						index = player.controls.index(event.key) #VLADYS CONTROLES ACTION SPACE
		# 					except:
		# 						pass
		# 					else:
		# 						if index == 0:
		# 							if player.fire() and play_sounds:
		# 								sounds["fire"].play()
		# 						elif index == 1:
		# 							player.pressed[0] = True
		# 						elif index == 2:
		# 							player.pressed[1] = True
		# 						elif index == 3:
		# 							player.pressed[2] = True
		# 						elif index == 4:
		# 							player.pressed[3] = True
		# 		elif event.type == pygame.KEYUP and not self.game_over and self.active:
		# 			for player in players:
		# 				if player.state == player.STATE_ALIVE:
		# 					try:
		# 						index = player.controls.index(event.key)
		# 					except:
		# 						pass
		# 					else:
		# 						if index == 1:
		# 							player.pressed[0] = False
		# 						elif index == 2:
		# 							player.pressed[1] = False
		# 						elif index == 3:
		# 							player.pressed[2] = False
		# 						elif index == 4:
		# 							player.pressed[3] = False

		# 	for player in players:
		# 		if player.state == player.STATE_ALIVE and not self.game_over and self.active:
		# 			if player.pressed[0] == True:
		# 				player.move(self.DIR_UP);
		# 			elif player.pressed[1] == True:
		# 				player.move(self.DIR_RIGHT);
		# 			elif player.pressed[2] == True:
		# 				player.move(self.DIR_DOWN);
		# 			elif player.pressed[3] == True:
		# 				player.move(self.DIR_LEFT);
		# 		player.update(time_passed)

		# 	for enemy in enemies:
		# 		if enemy.state == enemy.STATE_DEAD and not self.game_over and self.active:
		# 			enemies.remove(enemy)
		# 			if len(self.level.enemies_left) == 0 and len(enemies) == 0:
		# 				self.finishLevel()
		# 		else:
		# 			enemy.update(time_passed)

		# 	if not self.game_over and self.active:
		# 		for player in players:
		# 			if player.state == player.STATE_ALIVE:
		# 				if player.bonus != None and player.side == player.SIDE_PLAYER:
		# 					self.triggerBonus(bonus, player)
		# 					player.bonus = None
		# 			elif player.state == player.STATE_DEAD:
		# 				self.superpowers = 0
		# 				player.lives -= 1
		# 				if player.lives > 0:
		# 					self.respawnPlayer(player)
		# 				else:
		# 					self.gameOver()

		# 	for bullet in bullets:
		# 		if bullet.state == bullet.STATE_REMOVED:
		# 			bullets.remove(bullet)
		# 		else:
		# 			bullet.update()

		# 	for bonus in bonuses:
		# 		if bonus.active == False:
		# 			bonuses.remove(bonus)

		# 	for label in labels:
		# 		if not label.active:
		# 			labels.remove(label)

		# 	if not self.game_over:
		# 		if not castle.active:
		# 			self.gameOver()

		# 	gtimer.update(time_passed)

		# 	self.draw() #VLADYS RENDER
		# 	screen_array = pygame.surfarray.array3d(screen)
		# 	screen_array = np.transpose(screen_array, (1, 0, 2))
		# 	screen_array_grayscale = rgb_to_grayscale(screen_array)

		# 	# Display the array using matplotlib
		# 	# plt.imshow(screen_array_grayscale)
		# 	# plt.show()
  
	def get_mapinfo(self):
		global players, bullets
		mapinfo=[]
		mapinfo.append([])
		mapinfo.append([])
		mapinfo.append([])
		mapinfo.append([])
		for bullet in bullets:
			if bullet.owner == bullet.OWNER_ENEMY:
				nrect=bullet.rect.copy()
				mapinfo[0].append([nrect,bullet.direction,bullet.speed])
		for enemy in enemies:
			nrect=enemy.rect.copy()
			mapinfo[1].append([nrect,enemy.direction,enemy.speed,enemy.type])
		for tile in game.level.mapr:
			nrect=pygame.Rect(tile.left, tile.top, 16, 16)
			mapinfo[2].append([nrect,tile.type])
		for player in players:
			nrect=player.rect.copy()
			mapinfo[3].append([nrect,player.direction,player.speed,player.shielded])
		return mapinfo

'''
===============================================================================================================================
														RL TRAINING ENVIRONMENT
===============================================================================================================================
'''


class TanksEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} #TOCHECK

	def __init__(self, render_mode=None):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_distance_closest_enemy_to_castle, obs_distance_closest_enemy_to_player
		global obs_flag_castle_danger, obs_flag_enemy_in_line, obs_flag_bullet_avoidance_triggered, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot
		obs_distance_closest_enemy_to_castle = 0
		obs_distance_closest_enemy_to_player = 0
		obs_flag_castle_danger = 0
		obs_flag_enemy_in_line = 0
		obs_flag_bullet_avoidance_triggered = 0
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		obs_flag_hot = 0
		self.width = 208   # screen width
		self.height = 208  # screen height
		self.paso = 0

		# Initialize the heat map
		self.heat_map = np.zeros((13, 13))
		self.grid_size = 32
		self.grid_position = [0, 0]
		self.heat_decay_rate = 0.05  # Rate at which heat values decay each step
		self.heat_base_penalty = 0.01 # Penalty rate for staying on hot spots
		
		

		# Define the observation space for grayscale
		# self.observation_space = spaces.Box(low = 0, high = 255, shape = (width, height), dtype = np.uint8)
		self.frame_stack = deque(maxlen=4)
		empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
		for _ in range(4):
			self.frame_stack.append(empty_frame)
		#self.previous_frame = empty_frame
		#self.frames_ago4 = empty_frame
		#self.frames_ago8 = empty_frame
		#self.frames_ago16 = empty_frame
		#self.frames_ago32 = empty_frame

		# In case it is desired to add extra information, we should use a dictionary and CnnPolicy cant be used
		self.observation_space = spaces.Dict(
			{
				"obs_frames": spaces.Box(low=0, high=255, shape=(3, self.width, self.height), dtype=np.uint8),
				"prev_action": spaces.MultiDiscrete([2, 5]),
				"ai_bot_actions": spaces.MultiDiscrete([2, 5]),
				"flags": spaces.MultiBinary(6),
				"enemy_distance_to_castle": spaces.Discrete(850),
				"enemy_distance_to_player": spaces.Discrete(850),
				"enemies_left": spaces.Discrete(20),
				"temperature": spaces.Discrete(26),
				"player_position":spaces.MultiDiscrete([26, 26]),
				"player_lives": spaces.Discrete(4),

			}
		)

		# Define the action space: no move, shoot, move up, down, right and left           
		self.action_space = spaces.MultiDiscrete([2, 5])
		# Initialize variables (this was in the main part of tanks.py)
		gtimer = Timer()

		sprites = None
		screen = None
		screen_array = None
		screen_array_grayscale = empty_frame
		players = []
		enemies = []
		bullets = []
		bonuses = []
		labels = []

		play_sounds = False
		sounds = {}

		game = Game()
		castle = Castle()
		game.showMenu()
  
		# Initialize timer for efficiency bonus
		#self.level_start_time = None

  
	# def __init__(self, render_mode=None, size=5):
	#     self.size = size  # The size of the square grid
	#     self.window_size = 512  # The size of the PyGame window

	#     # Observations are dictionaries with the agent's and the target's location.
	#     # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
	#     self.observation_space = spaces.Dict(
	#         {
	#             "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
	#             "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
	#         }
	#     )

	#     # We have 4 actions, corresponding to "right", "up", "left", "down"
	#     self.action_space = spaces.Discrete(4)

	#     """
	#     The following dictionary maps abstract actions from `self.action_space` to
	#     the direction we will walk in if that action is taken.
	#     I.e. 0 corresponds to "right", 1 to "up" etc.
	#     """
	#     self._action_to_direction = {
	#         0: np.array([1, 0]),
	#         1: np.array([0, 1]),
	#         2: np.array([-1, 0]),
	#         3: np.array([0, -1]),
	#     }

	#     assert render_mode is None or render_mode in self.metadata["render_modes"]
	#     self.render_mode = render_mode

	#     """
	#     If human-rendering is used, `self.window` will be a reference
	#     to the window that we draw to. `self.clock` will be a clock that is used
	#     to ensure that the environment is rendered at the correct framerate in
	#     human-mode. They will remain `None` until human-mode is used for the
	#     first time.
	#     """
	#     self.window = None
	#     self.clock = None
	
	def _get_obs(self):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_distance_closest_enemy_to_castle, obs_distance_closest_enemy_to_player
		global obs_flag_castle_danger, obs_flag_enemy_in_line, obs_flag_bullet_avoidance_triggered, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot
		#return screen_array_grayscale
		# print(np.array(game.ai_bot_actions))
		# print(self.prev_action)
		# print(np.array([obs_flag_castle_danger, obs_flag_enemy_in_line, obs_flag_bullet_avoidance_triggered, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot]))
		# print(len(enemies))
		# print(int(self.heat_map[self.grid_position[0], self.grid_position[1]]))
		# print(np.array(self.grid_position))
		# print(players[0].lives)
		return {
			"obs_frames": self.obs_frames,
			"ai_bot_actions": np.array(game.ai_bot_actions),
			"prev_action": self.prev_action,
			"flags": np.array([obs_flag_castle_danger, obs_flag_enemy_in_line, obs_flag_bullet_avoidance_triggered, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot]),
			"enemy_distance_to_castle": obs_distance_closest_enemy_to_castle,
			"enemy_distance_to_player": obs_distance_closest_enemy_to_player,
			"enemies_left": len(enemies),
			"temperature": int(self.heat_map[self.grid_position[0], self.grid_position[1]]),
			"player_position": np.array(self.grid_position),
			"player_lives": players[0].lives,
		}
		# return {"agent": self._agent_location, "target": self._target_location}
	
	def _get_info(self):
		return {"Info": 0}
		# return {
		#     "distance": np.linalg.norm(
		#         self._agent_location - self._target_location, ord=1
		#     )
		# }

	def kill_ai_process(self, p):
		os.kill(p.pid, 9)
		#print("Killed AI Process!")

	def clear_queue(self, queue):
		# Use a loop to clear the queue instead of a single get
		while not queue.empty():
			try:
				queue.get(False)
			except Empty:  # Catch the correct exception
				break  # Exit loop if queue is empty
	
	def reset(self, seed=None, options=None):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_castle_danger, obs_flag_enemy_in_line, obs_flag_bullet_avoidance_triggered, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot
		# Reset the timer at the start of a level
		#self.level_start_time = time.time()
		self.reward = 0
		self.paso = 0
		self.prev_action = np.array([0, 4])
		#self.killed_enemies = 0


		obs_flag_castle_danger = 0
		obs_flag_enemy_in_line = 0
		obs_flag_bullet_avoidance_triggered = 0
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		obs_flag_hot = 0

		empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
		# Reset the frame stack
		self.frame_stack.clear()  # Clear existing frames
		for _ in range(4):
			self.frame_stack.append(empty_frame)
		#self.frames_ago4 = empty_frame
		#self.frames_ago8 = empty_frame
		#self.frames_ago16 = empty_frame
		#self.frames_ago32 = empty_frame
		#self.previous_frame = empty_frame
		self.obs_frames = np.stack([empty_frame, empty_frame, empty_frame], axis=0)
		self.heat_map = np.zeros((13, 13))
		#game.gameOver()
		game.nextLevel()
		#game.finishLevel()
		game.ai_bot_actions = [0 if x is None else x for x in game.ai_bot_actions]
		observation = self._get_obs()
		info = self._get_info()
  
		return observation, info

		# # We need the following line to seed self.np_random
		# super().reset(seed=seed)

		# # Choose the agent's location uniformly at random
		# self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

		# # We will sample the target's location randomly until it does not coincide with the agent's location
		# self._target_location = self._agent_location
		# while np.array_equal(self._target_location, self._agent_location):
		#     self._target_location = self.np_random.integers(
		#         0, self.size, size=2, dtype=int
		#     )

		# observation = self._get_obs()
		# info = self._get_info()

		# if self.render_mode == "human":
		#     self._render_frame()

		# return observation, info

	def step(self, action):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_stupid, obs_flag_player_collision, obs_flag_hot
		self.reward = 0
		#danger_flag = 0
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		time_passed = 20
		self.paso += 1
  
		# Update the info of the map and get AI bot actions
		mapinfo = game.get_mapinfo()
		if game.p_mapinfo.empty() == True:
			game.p_mapinfo.put(mapinfo)
		if game.c_control.empty() != True:
			try:
				game.ai_bot_actions = game.c_control.get(False)
			except queue.empty:
				skip_this = True

		#for i in range(4): #FRAME SKIPPING
		# Constants representing directions
		DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = range(4)
		#print(self.prev_action, action)
		#time.sleep(1/100)
		pygame.event.pump()
		for player in players:
			if player.state == player.STATE_ALIVE and not game.game_over and game.active:
				if action[0] == 1:
					player.fire()
					#self.reward += 0.0001
				if action[1] == 0:
					player.move(game.DIR_UP)
					if self.prev_action[1] == 0 and obs_flag_player_collision == 0:
						self.reward += 0.05
					# if self.prev_action != 2 and self.prev_action != 0:
					# 	self.reward -= 0.005
					# else:
					# 	self.reward += 0.005
				if action[1] == 1:
					player.move(game.DIR_RIGHT)
					if self.prev_action[1] == 1 and obs_flag_player_collision == 0:
						self.reward += 0.05
					# if self.prev_action != 3 and self.prev_action != 0:
					# 	self.reward -= 0.005
					# else:
					# 	self.reward += 0.005
				if action[1] == 2:
					player.move(game.DIR_DOWN)
					if self.prev_action[1] == 2 and obs_flag_player_collision == 0:
						self.reward += 0.05
					# if self.prev_action != 4 and self.prev_action != 0:
					# 	self.reward -= 0.005
					# else:
					# 	self.reward += 0.005
				if action[1] == 3:
					player.move(game.DIR_LEFT)
					if self.prev_action[1] == 3 and obs_flag_player_collision == 0:
						self.reward += 0.05
					# if self.prev_action != 5 and self.prev_action != 0:
					# 	self.reward -= 0.005
					# else:
					# 	self.reward += 0.005
			# 	elif action[1] == 4:
			# 		self.reward -= 0.2
					
			self.prev_action = action
			if action[0] == game.ai_bot_actions[0] and action[0] == 1:
				self.reward += 0.2

			if action[1] == game.ai_bot_actions[1] and action[1] != 4:
				self.reward += 0.1
			
			# Get the player's current grid position
			self.grid_position = (player.rect.centerx // self.grid_size, player.rect.centery // self.grid_size)
			
			# Increase the heat value of the current position
			if self.heat_map[self.grid_position] < 25:
				self.heat_map[self.grid_position] += 1
			
			if self.heat_map[self.grid_position]	> 9:
				obs_flag_hot = 1
			else:
				obs_flag_hot = 0

			# Apply a negative reward based on the heat value
			self.reward -= self.heat_base_penalty * (1.22 ** self.heat_map[self.grid_position])

			# Decay the heat map
			self.heat_map *= (1 - self.heat_decay_rate)
			#print(np.round(self.heat_map, 1))

			player.update(time_passed)

		for enemy in enemies:
			if enemy.state == enemy.STATE_DEAD and not game.game_over and game.active:
				self.reward += 2 # RW KILL
				#self.killed_enemies += 1
				
				#print(len(enemies))
				#print("+50 for killing a tank!", self.reward)
				enemies.remove(enemy)
				#print(len(enemies))
				#self.reward += 20 * ((6 / (len(enemies) + 1)) - 1) # RW TANKS LEFT
				#print(f"+{10 * ((6 / (len(enemies) + 1)) - 1)} for the remaining tanks left!", self.reward)
				if len(game.level.enemies_left) == 0 and len(enemies) == 0:
					self.reward += 10 # RW WIN
					#print("+100 for winning the game!", self.reward)
					# level_time = time.time() - self.level_start_time
					# if level_time >= 150:
					# 	self.reward -= 100 # RW TIME ELLAPSES
					# 	#print("-20 for exceeding the maximum time!", self.reward)
					# else:
					# 	self.reward += 100 / (level_time + 1) # RW TIME EFFICIENCY
					print("You killed all enemy tanks! :). Reward: ", self.reward)
					self.kill_ai_process(game.p)
					self.clear_queue(game.p_mapinfo)
					self.clear_queue(game.c_control)
					game.game_over = 1
			else:
				enemy.update(time_passed)

		if not game.game_over and game.active:
			for player in players:
				if player.state == player.STATE_ALIVE:
					if player.bonus != None and player.side == player.SIDE_PLAYER:
						game.triggerBonus(player.bonus, player)
						self.reward += 1 # RW BONUS
						player.bonus = None
				elif player.state == player.STATE_DEAD:
					self.reward -= 4 # RW DEAD
					#print("-50 for dying! ", self.reward)
					game.superpowers = 0
					player.lives -= 1
					if player.lives > 0:
						game.respawnPlayer(player)
					else:
						self.reward -= 6
						print("You died! :(. Reward: ", self.reward)
						self.kill_ai_process(game.p)
						self.clear_queue(game.p_mapinfo)
						self.clear_queue(game.c_control)
						game.game_over = 1

		for bullet in bullets:
			if bullet.state == bullet.STATE_REMOVED:
				bullets.remove(bullet)
			else:
				bullet.update()

		# 	if bullet.owner == Bullet.OWNER_ENEMY:  # We consider enemy bullets as danger
		# 		# Check if the bullet is moving towards the player's tank
		# 		if bullet.direction == DIR_DOWN and bullet.rect.bottom < players[0].rect.top and bullet.rect.left <= players[0].rect.right and bullet.rect.right >= players[0].rect.left:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_UP and bullet.rect.top > players[0].rect.bottom and bullet.rect.left <= players[0].rect.right and bullet.rect.right >= players[0].rect.left:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_RIGHT and bullet.rect.right < players[0].rect.left and bullet.rect.top <= players[0].rect.bottom and bullet.rect.bottom >= players[0].rect.top:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_LEFT and bullet.rect.left > players[0].rect.right and bullet.rect.top <= players[0].rect.bottom and bullet.rect.bottom >= players[0].rect.top:
		# 			danger_flag = 1

		# 		# Check if the bullet is moving towards the base
		# 		if bullet.direction == DIR_DOWN and bullet.rect.bottom < castle.rect.top and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_UP and bullet.rect.top > castle.rect.bottom and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_RIGHT and bullet.rect.right < castle.rect.left and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
		# 			danger_flag = 1
		# 		if bullet.direction == DIR_LEFT and bullet.rect.left > castle.rect.right and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
		# 			danger_flag = 1


			if bullet.owner == Bullet.OWNER_PLAYER:  # We consider suicide as danger
				# Check if the bullet is moving towards the base
				if bullet.direction == DIR_DOWN and bullet.rect.bottom < castle.rect.top and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
					obs_flag_stupid = 1
				if bullet.direction == DIR_UP and bullet.rect.top > castle.rect.bottom and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
					obs_flag_stupid = 1
				if bullet.direction == DIR_RIGHT and bullet.rect.right < castle.rect.left and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
					obs_flag_stupid = 1
				if bullet.direction == DIR_LEFT and bullet.rect.left > castle.rect.right and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
					obs_flag_stupid = 1

		# if danger_flag == 1:
		# 	self.reward -= 0.008
		
		if obs_flag_stupid == 1:
			self.reward -= 0.1
			
			

		for bonus in bonuses:
			if bonus.active == False:
				bonuses.remove(bonus)

		for label in labels:
			if not label.active:
				labels.remove(label)

		if not game.game_over:
			if not castle.active:
				self.reward -= 10 # RW LOST
				#print("Castle not active! Reward: ", self.reward)
				self.kill_ai_process(game.p)
				self.clear_queue(game.p_mapinfo)
				self.clear_queue(game.c_control)
				game.game_over = 1

		gtimer.update(time_passed)
		#self.reward -= 0.001 # RW TIMESTEP

			
		
		game.draw() #VLADYS RENDER

		# Update the observation with the new current frame
		screen_array = pygame.surfarray.array3d(screen)
		screen_array = np.transpose(screen_array, (1, 0, 2))
		screen_array_grayscale = rgb_to_grayscale(screen_array)

		# Add the new frame to the stack and update the specific frames
		self.frame_stack.append(screen_array_grayscale)
		#self.previous_frame = self.frame_stack[-2]
		#self.frames_ago4 = self.frame_stack[-4]
		#self.frames_ago8 = self.frame_stack[-8]
		#self.frames_ago16 = self.frame_stack[-16]
		#self.frames_ago32 = self.frame_stack[-32]
		
		# print("test : ", np.array(game.ai_bot_actions))
		# print("test : ", action)
		
		self.obs_frames = np.stack([screen_array_grayscale, self.frame_stack[-2], self.frame_stack[-4]], axis=0)
		game.ai_bot_actions = [0 if x is None else x for x in game.ai_bot_actions]
		observation = self._get_obs()

		####################  DEBUGGING OBSERVATION  ###################################
		# # Assuming observation is your dictionary with 'current_frame' and 'previous_frame'

		# # Set up the matplotlib figure and axes
		# fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns for current and previous frames

		# # Display current frame
		# axs[0].imshow(observation['current_frame'])
		# axs[0].set_title('Current Frame')
		# axs[0].axis('off')  # Hide axes for better visualization

		# # Display previous frame
		# axs[1].imshow(observation['previous_frame'])
		# axs[1].set_title('Previous Frame')
		# axs[1].axis('off')  # Hide axes for better visualization

		# plt.show()  # Show the image window
		###############################################################################

		#self.reward += self.killed_enemies*0.001
		reward = self.reward
		terminated = game.game_over
		truncated = False
		# if self.paso == 2999 and terminated != 1:
		# 	truncated = True
		info = self._get_info()
		#print("step: ", self.paso,"  ","danger flag: ", danger_flag,"  ", "stupid flag: ", obs_flag_stupid,"  ", "reward: ", reward) #DEBUGGING
		#print("step: ", self.paso,". RL Agent: ", action, ". AI Training Bot: ", game.ai_bot_actions, ". Reward: ", reward)
		return observation, reward, terminated, truncated, info		

		# # Map the action (element of {0,1,2,3}) to the direction we walk in
		# direction = self._action_to_direction[action]
		# # We use `np.clip` to make sure we don't leave the grid
		# self._agent_location = np.clip(
		#     self._agent_location + direction, 0, self.size - 1
		# )
		# # An episode is done iff the agent has reached the target
		# terminated = np.array_equal(self._agent_location, self._target_location)
		# reward = 1 if terminated else 0  # Binary sparse rewards
		# observation = self._get_obs()
		# info = self._get_info()

		# if self.render_mode == "human":
		#     self._render_frame()

		# return observation, reward, terminated, False, info
	
	def render(self):
		pass
		# if self.render_mode == "rgb_array":
		#     return self._render_frame()

	def _render_frame(self):
		pass
		# if self.window is None and self.render_mode == "human":
		#     pygame.init()
		#     pygame.display.init()
		#     self.window = pygame.display.set_mode(
		#         (self.window_size, self.window_size)
		#     )
		# if self.clock is None and self.render_mode == "human":
		#     self.clock = pygame.time.Clock()

		# canvas = pygame.Surface((self.window_size, self.window_size))
		# canvas.fill((255, 255, 255))
		# pix_square_size = (
		#     self.window_size / self.size
		# )  # The size of a single grid square in pixels

		# # First we draw the target
		# pygame.draw.rect(
		#     canvas,
		#     (255, 0, 0),
		#     pygame.Rect(
		#         pix_square_size * self._target_location,
		#         (pix_square_size, pix_square_size),
		#     ),
		# )
		# # Now we draw the agent
		# pygame.draw.circle(
		#     canvas,
		#     (0, 0, 255),
		#     (self._agent_location + 0.5) * pix_square_size,
		#     pix_square_size / 3,
		# )

		# # Finally, add some gridlines
		# for x in range(self.size + 1):
		#     pygame.draw.line(
		#         canvas,
		#         0,
		#         (0, pix_square_size * x),
		#         (self.window_size, pix_square_size * x),
		#         width=3,
		#     )
		#     pygame.draw.line(
		#         canvas,
		#         0,
		#         (pix_square_size * x, 0),
		#         (pix_square_size * x, self.window_size),
		#         width=3,
		#     )

		# if self.render_mode == "human":
		#     # The following line copies our drawings from `canvas` to the visible window
		#     self.window.blit(canvas, canvas.get_rect())
		#     pygame.event.pump()
		#     pygame.display.update()

		#     # We need to ensure that human-rendering occurs at the predefined framerate.
		#     # The following line will automatically add a delay to keep the framerate stable.
		#     self.clock.tick(self.metadata["render_fps"])
		# else:  # rgb_array
		#     return np.transpose(
		#         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
		#     )
	
	def close(self):
		pass
		# if self.window is not None:
		#     pygame.display.quit()
		#     pygame.quit()
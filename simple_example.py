import numpy as np
import math
import sys
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

# Default recursion limit is 1000, which is not enough for recursion based SCC calculation
sys.setrecursionlimit(10**6)
# Maximum number of iterations for generating trajectory
MAX_NUM_ITER = 2000
# Font size for plotting
FS = 16
label_dtype = np.uint8
obs_label = np.iinfo(label_dtype).max

# Set up data from carAbst.cpp in ROCS
state_space = [[0,10],
			   [0,10]]
nominal_grid_size = [0.2,0.2]

# lower bounds for all dim, then upper bound for all dim (different from ROCS)
obstacle = [[[1.6,4.0],[5.7,5.0]],
			[[3.0,5.0],[5.0,8.0]],
			[[4.3,1.8],[5.7,4.0]],
			[[5.7,1.8],[8.5,2.5]]]

goals = [[[1.0,0.5],[2.0,2.0]],
		 [[0.5,7.5],[2.5,8.5]],
		 [[7.1,4.6],[9.1,6.4]]]


#Generated using hoa from spot. The LTL is F(a1 & F(a2 & F(a3 & (!a2 U a1))))
# dba_lookup_table = [[0, 0, 0, 0, 0, 0, 0, 0],
# 				    [1, 0, 2, 0, 1, 0, 2, 0],
# 				    [2, 2, 2, 2, 1, 0, 2, 0],
# 				    [3, 4, 3, 2, 3, 4, 3, 0],
# 				    [4, 4, 2, 2, 4, 4, 2, 0]]
# dba_init_state_idx = 3
# dba_acc_state_idx = 0

dba_lookup_table = [[1, 1, 1, 1, 2, 2, 3, 0],
				    [1, 1, 1, 1, 2, 2, 3, 0],
				    [2, 2, 3, 0, 2, 2, 3, 0],
				    [3, 0, 3, 0, 3, 0, 3, 0]]
dba_init_state_idx = 0
dba_acc_state_idx = 0

# Keep original copy for plotting
obstacle_original = obstacle.copy()
goals_original = goals.copy()

# reversing order to ensure traversal order is xyz instead of zyx
state_space.reverse()
nominal_grid_size.reverse()
obstacle = [[[b for b in reversed(a[0])],[b for b in reversed(a[1])]] for a in obstacle]
goals = [[[b for b in reversed(a[0])],[b for b in reversed(a[1])]] for a in goals]

dim = len(state_space)
num_ap = len(goals)
num_label = len(dba_lookup_table[0])
num_dba_state = len(dba_lookup_table)
dba_lookup_table = np.array(dba_lookup_table)

num_grid = [math.floor((x[1]-x[0])/y) for x,y in zip(state_space, nominal_grid_size)]
actual_grid_size = [(x[1]-x[0])/y for x,y in zip(state_space, num_grid)]
label_mat = np.zeros(num_grid,dtype=label_dtype)
decode_lookup = [0]*(dim+1)
decode_lookup[0] = math.prod(num_grid)
num_product_state = num_dba_state*math.prod(num_grid)
for i in range(1,dim+1):
	decode_lookup[i] = decode_lookup[i-1]//num_grid[i-1]
# Goal first, obstacle second to ensure overlap area are marked as obstacle
# Floor and ceiling usage needs to be checked here
# Label goal
for i, (lb,ub) in enumerate(goals):
	val = 1 << i
	lb = [math.ceil((a-b[0])/c) for a,b,c in zip(lb, state_space, actual_grid_size)]
	ub = [math.floor((a-b[0])/c) for a,b,c in zip(ub, state_space, actual_grid_size)]
	label_mat[*[slice(a,b) for a,b in zip(lb,ub)]] += val

# Label obstacle
obstacle_idx = [[[math.floor((a-b[0])/c) for a,b,c in zip(lb, state_space, actual_grid_size)],
				 [math.ceil((a-b[0])/c) for a,b,c in zip(ub, state_space, actual_grid_size)]] 
				 for lb, ub in obstacle]
for x,y in obstacle_idx:
	label_mat[*[slice(a,b) for a,b in zip(x,y)]] = obs_label

# Save labels as txt file
np.savetxt("label_mat.txt",label_mat.astype(int),fmt="%i",delimiter=',')

# Find all neighbor on the product space (no diagonals on state space). 
# Input length is dim+1 where the first element is the current dba state.
def find_post(idx_list):
	result = []
	cur_dba_state = idx_list[0]
	cur_state_idx = idx_list[1:]
	cur_label = dba_lookup_table[cur_dba_state][label_mat[*idx_list[1:]]]
	for i in range(dim):
		cur_state_idx[i] -= 1
		if cur_state_idx[i] >= 0 and label_mat[*cur_state_idx] != obs_label:
			result.append([cur_label, *cur_state_idx])

		cur_state_idx[i] += 2
		if cur_state_idx[i] != label_mat.shape[i] and label_mat[*cur_state_idx] != obs_label:
			result.append([cur_label, *cur_state_idx])	

		cur_state_idx[i] -= 1
	return result

def find_pre(idx_list):
	result = []
	cur_dba_state = idx_list[0]
	cur_state_idx = idx_list[1:]
	# for i in [1,-1]:
	# 	for d in range(dim):
	# 		cur_state_idx[d] += i
	# 		if cur_state_idx[d] >= 0 and cur_state_idx[d] != label_mat.shape[d] and label_mat[*cur_state_idx] != obs_label:
	# 			pre_dba_state = np.argwhere(dba_lookup_table[:,label_mat[*cur_state_idx]] == cur_dba_state).flatten()
	# 			for q in pre_dba_state: 
	# 				result.append([q, *cur_state_idx])
	# 		cur_state_idx[d] -= i
	for i in range(dim-1,-1,-1):
		cur_state_idx[i] -= 1
		if cur_state_idx[i] >= 0 and label_mat[*cur_state_idx] != obs_label:
			pre_dba_state = np.argwhere(dba_lookup_table[:,label_mat[*cur_state_idx]] == cur_dba_state).flatten()
			for q in pre_dba_state: 
				result.append([q, *cur_state_idx])

		cur_state_idx[i] += 2
		if cur_state_idx[i] != label_mat.shape[i] and label_mat[*cur_state_idx] != obs_label:
			pre_dba_state = np.argwhere(dba_lookup_table[:,label_mat[*cur_state_idx]] == cur_dba_state).flatten()
			for q in pre_dba_state: 
				result.append([q, *cur_state_idx])

		cur_state_idx[i] -= 1
	return result

def decode_idx(idx):
	result = [0]*(dim+1)
	cur_idx = idx
	for i,val in enumerate(decode_lookup):
		result[i], cur_idx = divmod(cur_idx,val)
	return result

def encode_idx(idx_list):
	return sum([x*y for x,y in zip(idx_list, decode_lookup)])


# Algorithm reference from Tarjan's strongly connected components algorithm on Wikipedia
# This function modified the original algorithm to only find nontrivial SCC with dba acc states
def tarjan_SCC():
	cur_idx = 0
	stack = []
	on_stack_flag = np.zeros((num_product_state,))
	# column 0: index, column 1: lowlink
	idx_mat = np.zeros((num_product_state, 2))
	idx_mat[:,0] = -1
	idx_mat[:,1] = 0
	num_scc = 0
	scc_acc_idx = set()
	def strongconnect(v):
		nonlocal cur_idx, num_scc, scc_acc_idx
		idx_mat[v,0] = cur_idx
		idx_mat[v,1] = cur_idx
		cur_idx += 1
		stack.append(v)
		on_stack_flag[v] = 1
		neighbors = [encode_idx(x) for x in find_post(decode_idx(v))]
		for w in neighbors:
			if idx_mat[w,0] < 0:
				strongconnect(w)
				idx_mat[v,1] = min(idx_mat[v,1],idx_mat[w,1])
			elif on_stack_flag[w]:
				idx_mat[v,1] = min(idx_mat[v,1], idx_mat[w,0])
		if idx_mat[v,1] == idx_mat[v,0]:
			num_scc += 1
			scc_list = []
			while True:
				w = stack.pop()
				scc_list.append(w)
				w_dba_state = decode_idx(w)[0]
				if w_dba_state == dba_acc_state_idx:
					scc_acc_idx.add(w)
				on_stack_flag[w] = False
				if w == v:
					break

	for i in range(num_product_state):
		# ignore obstacle states
		idx_list = decode_idx(i)
		if label_mat[*idx_list[1:]] == obs_label:
			continue
		if idx_mat[i,0] < 0:
			strongconnect(i)
	return num_scc, scc_acc_idx

result = tarjan_SCC() 
idx_queue = deque(result[1])
# column 0: visited, column 1: strategy
strategy_mat = np.zeros([num_product_state,2],dtype=np.int64)
bfs_counter = 0
while len(idx_queue) > 0:
	cur_idx = idx_queue.popleft()
	pre_states = find_pre(decode_idx(cur_idx))
	for p_list in pre_states:
		p = encode_idx(p_list)
		if not strategy_mat[p,0]:
			strategy_mat[p,0] = 1
			strategy_mat[p,1] = cur_idx
			if p_list[0] != dba_acc_state_idx:
				idx_queue.append(p)	
winning_set = np.argwhere(strategy_mat[:,0]==1).flatten()
winning_set = np.array([decode_idx(x) for x in winning_set])
winning_set = winning_set[winning_set[:,0]==dba_init_state_idx]
# Plotting trajectory using random initial state from winning set
init_idx = winning_set[np.random.randint(0,winning_set.shape[0])].tolist()
cur_idx = encode_idx(init_idx)

def idx_to_coord(idx_list):
	return [(2*x+1)*y*0.5+z[0] for x,y,z in zip(idx_list,actual_grid_size,state_space)]

init_coord = idx_to_coord(init_idx[1:])
traj_mat = np.zeros((MAX_NUM_ITER+1,dim))
traj_mat[0] = init_coord
num_acc_visit = 0
for i in range(MAX_NUM_ITER):
	next_idx = decode_idx(strategy_mat[cur_idx,1])
	if next_idx[0] == dba_acc_state_idx:
		num_acc_visit += 1
	traj_mat[i+1] = idx_to_coord(next_idx[1:])
	cur_idx = strategy_mat[cur_idx,1]

traj_mat = np.fliplr(traj_mat)

print(f"Number of acc visit: {num_acc_visit}")
plt.figure()
fig, ax = plt.subplots()
ax.set_xticks(np.arange(0,10,0.2))
ax.set_yticks(np.arange(0,10,0.2))
for lb,ub in goals_original:
	ax.add_patch(Rectangle(lb,ub[0]-lb[0],ub[1]-lb[1],facecolor='gold'))
for lb,ub in obstacle_original:
	ax.add_patch(Rectangle(lb,ub[0]-lb[0],ub[1]-lb[1],facecolor='dimgray'))
plt.text(1.4, 1.2, '$a_1$', fontsize=FS)
plt.text(1.3, 8.0, '$a_2$', fontsize=FS)
plt.text(8.0, 5.5, '$a_3$', fontsize=FS)
plt.xlim(state_space[0])
plt.ylim(state_space[1])
plt.plot(traj_mat[:,0],traj_mat[:,1])
font = FontProperties()
font.set_weight('bold')
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.plot(traj_mat[0,0], traj_mat[0,1], marker='^', markerfacecolor='r')
plt.plot(traj_mat[-1,0], traj_mat[-1,1], marker='v', markerfacecolor='g')
plt.grid()
plt.savefig(f"./simple_example.png",dpi=1000)

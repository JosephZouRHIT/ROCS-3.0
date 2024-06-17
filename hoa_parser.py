from re import split
import numpy as np
import itertools
import regex

keyword_lookup = ["States:", "Start:", "AP:", "Acceptance:", "--BODY--"]

file_name = "./generalized_buchi.hoa"
with open(file_name, "r") as fp:
	lines = fp.readlines()

# Helper function to split lines starting in "State:"
def split_state(input_str):
	reg_extract = regex.compile(r'(?:(\{(?>[^\{\}]+|(?1))*\})|\S)+')
	res = [x.group() for x in reg_extract.finditer(input_str)]
	return res

# Helper function to split lines starting in "State:"
def split_transition_label(input_str):
	reg_extract = regex.compile(r'(?:(\[(?>[^\[\]]+|(?1))*\])|\S)+')
	res = [x.group() for x in reg_extract.finditer(input_str)]
	return res

#TODO: add accepting group mapping later. 
#Since we only consider DBA at the moment, there is only one accepting state.
def body_parser(body_lines, num_states, num_ap):

	lookup_table = np.zeros((num_states, 1 << num_ap))
	cur_state_idx = -1
	acc_state_idx = -1
	for line in body_lines:
		if line[:6] == "State:":
			cur_state_idx += 1
			split_result = split_state(line)
			# Add mode logic in this block to accomodate multiple acc states or groups.
			if len(split_result) > 2:
				acc_state_idx = cur_state_idx
		else:
			split_result = split_transition_label(line)
			val = int(split_result[1])
			if split_result[0] == "[t]":
				lookup_table[cur_state_idx,:] = val
				continue
			or_split_result = split_result[0][1:-1].split(" | ")
			for cond in or_split_result:
				split_result = cond.split('&')
				remaining_idx = list(range(num_ap))
				label_base = 0
				for p in split_result:
					if p[0] == '!':
						idx = int(p[1:])
						remaining_idx.remove(idx)
						continue
					idx = int(p)
					remaining_idx.remove(idx)
					label_base += 1 << idx
				idx_comb = [[0,1]] * len(remaining_idx)
				idx_comb = itertools.product(*idx_comb)
				for c in idx_comb:
					cur_label = label_base
					for i in range(len(remaining_idx)):
						cur_label += c[i] << remaining_idx[i]
					lookup_table[cur_state_idx,cur_label] = val

	return acc_state_idx, lookup_table



section_idx = 0
num_states = 0
start_idx = 0
num_ap = 0
num_acc_grp = 0
dba_mat = None
acc_state_idx = -1
for i,l in enumerate(lines):
	words = l.split()
	if words[0] != keyword_lookup[section_idx]:
		continue
	match section_idx:
		case 0:
			num_states = int(words[1])
			section_idx += 1
		case 1:
			start_idx = int(words[1])
			section_idx += 1
		case 2:
			num_ap = int(words[1])
			section_idx += 1
		case 3:
			num_acc_grp = int(words[1])
			section_idx += 1
		case 4:
			acc_state_idx, dba_mat = body_parser(lines[i+1:-1], num_states, num_ap)
			break

print(f"num_states:{num_states}")
print(f"start_idx:{start_idx}")
print(f"acc_state_idx:{acc_state_idx}")
print(f"lookup_table:{dba_mat}")
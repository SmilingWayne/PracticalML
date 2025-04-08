import bisect
from collections import defaultdict

def get_square_residue(x):
    s = 1
    i = 2
    while i * i <= x:
        exponent = 0
        while x % i == 0:
            exponent += 1
            x = x // i
        if exponent % 2 == 1:
            s *= i
        i += 1
    if x > 1:
        s *= x
    return s

n = int(input())
a = list(map(int, input().split()))
s_list = [get_square_residue(x) for x in a]

s_indices = defaultdict(list)
for idx, s in enumerate(s_list):
    s_indices[s].append(idx)

# Case 1: Three 1's
if len(s_indices.get(1, [])) >= 3:
    indices = s_indices[1][:3]
    print(f"{indices[0]+1} {indices[1]+1} {indices[2]+1}")
    exit()

# Case 2: Two same s and one 1
candidates = [s_val for s_val, indices in s_indices.items() if len(indices) >= 2]
for s_val in candidates:
    if s_val == 1:
        continue
    indices = s_indices[s_val]
    if len(indices) >= 2:
        idx_i = indices[0]
        idx_j = indices[1]
        if 1 in s_indices:
            one_indices = s_indices[1]
            pos = bisect.bisect_right(one_indices, idx_j)
            if pos < len(one_indices):
                idx_k = one_indices[pos]
                print(f"{idx_i+1} {idx_j+1} {idx_k+1}")
                exit()

# Case 3: Find i < j < k where s_i * s_j's residue equals s_k
for i in range(n):
    for j in range(i + 1, n):
        s_i = s_list[i]
        s_j = s_list[j]
        product = s_i * s_j
        target_s = get_square_residue(product)
        if target_s in s_indices:
            candidates_k = s_indices[target_s]
            pos = bisect.bisect_right(candidates_k, j)
            if pos < len(candidates_k):
                k = candidates_k[pos]
                print(f"{i+1} {j+1} {k+1}")
                exit()

print(-1)
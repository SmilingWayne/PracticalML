
import math


line1 = input().strip().split()
n = int(line1[0])
x = int(line1[1])
speed = list(map(int, input().strip().split()))
shapes = dict()
result = 0
record_close_0 = []
for i in range(n):
    num = int(input())
    temp = []
    prev = None 
    min_to_zero = 999999
    close_0 = []
    for idx in range(num):
        dist = 0
        curr = list(map(int, input().strip().split()))
        to_zero = math.sqrt(curr[0] ** 2 + curr[1] ** 2)
        if to_zero < min_to_zero:
            min_to_zero = to_zero 
            close_0 = [curr[0], curr[1]]
        if prev:
            dist += math.sqrt( (curr[0] - prev[0]) ** 2 +  (curr[1] - prev[1]) ** 2 )
        prev = [curr[0], curr[1]]
        temp.append(curr) 
        if idx == num - 1:
            dist += math.sqrt( (curr[0] - temp[0][0]) ** 2 +  (curr[1] - temp[0][1]) ** 2 )
        result += dist / speed[i]
    record_close_0.append(close_0)
    shapes[i] = temp

record_further_dist = 0

print(result)

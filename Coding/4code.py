import math

def min_prime_factor(n):
    if n % 2 == 0:
        return 2
    max_divisor = int(math.sqrt(n)) + 1
    for i in range(3, max_divisor, 2):
        if n % i == 0:
            return i
    return n

t = int(input())
arr = []
for _ in range(t):
    n = int(input())
    arr.append(n)

for i in arr:
    print(min_prime_factor(i))
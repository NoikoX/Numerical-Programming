import numpy as np
import math

# ttf first assignment will be written here

def first_norm(line):
    return sum(map(abs, line))

def second_norm(line):
    return math.sqrt(sum(n*n for n in line))

def infinity_norm(line):
    return max(map(abs, line))

def p_norm(line, p):
    return sum(abs(n) ** p for n in line) ** (1/p)


s = input("Enter a vector, just numbers separated with spaces: ")
vector = list(map(int, s.split()))
print(round(first_norm(vector), 6))
print(round(second_norm(vector), 6))
print(round(infinity_norm(vector), 6))
print(round(p_norm(vector, 10), 6))

print("Using built in methods below")


print(
round(np.linalg.norm(vector, ord=1), 6),
round(np.linalg.norm(vector, ord=2), 6),
round(np.linalg.norm(vector, ord=  10), 6), # here whatever will be instead of 3 it will work as p norm
round(np.linalg.norm(vector, ord=np.inf), 6)
)
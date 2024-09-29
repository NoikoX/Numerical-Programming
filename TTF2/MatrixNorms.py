import numpy as np
import random

"""
Task is like that:
input is 3x3 matrix, then we have to generate 10 random 3x3 matrices,
calculate differences between those matrices and our input matrix,
use 3 different norms on those differences,
and then check which norm is giving the smalles and largest output
so that then we can take a look and see which random matrix was close to our input matrix in some norm and etc.
"""


def frobenius_norm(matrix1, matrix2):
    difference = matrix1 - matrix2
    return np.sqrt(sum(sum(difference[i][j] ** 2 for j in range(3)) for i in range(3)))


def first_matrix_norm(matrix1, matrix2): # max col summ
    difference = np.abs(matrix1 - matrix2)
    max_col_sum = 0
    for col in range(3):
        col_sum = sum(difference[row][col] for row in range(3))
        max_col_sum = max(max_col_sum, col_sum)
    return max_col_sum


def infinity_matrix_norm(matrix1, matrix2): # max row summ
    difference = np.abs(matrix1 - matrix2)
    max_row_sum = 0
    for row in range(3):
        row_sum = sum(difference[row][col] for col in range(3))
        max_row_sum = max(max_row_sum, row_sum)
    return max_row_sum

    
def generate_random_matrix():
    return np.array([[random.randint(-10, 10) for _ in range(3)] for _ in range(3)])


def get_user_matrix():
    print("Enter a 3x3 matrix:")
    inputa = []
    for i in range(3):
        row = list(map(int, input(f"Row {i+1} (enter 3 integers separeteed with spaces pleaseeeee): ").split()))
        inputa.append(row)
    return np.array(inputa)





def main():
    # exit(0) I was testing sum stuff here
    
    user_matrix = get_user_matrix()

    print("User matrixx looks like that")
    print(user_matrix)
    
    random_matrices = [generate_random_matrix() for _ in range(10)]
    
    
    print("So here I print all the random matrices that I generated: ")
    
    for i, matrix in enumerate(random_matrices):
        print(i + 1)
        print(matrix)
        
        
    
    # here I create lists where I store the result, index of the random array and norm of the difference of random matrix an d user matrix
    
    frobenius_distances = []
    first_norm_distances = []
    infinity_norm_distances = []
    # in those as i see there are floats stored like [(6, np.float64(15.24.......)), ....]

    # here I calculate distances for each random matrix
    for i, random_matrix in enumerate(random_matrices):
        f_norm = frobenius_norm(user_matrix, random_matrix)
        frobenius_distances.append((i, f_norm))
        
        first_norm = first_matrix_norm(user_matrix, random_matrix)
        first_norm_distances.append((i, first_norm))
        
        inf_norm = infinity_matrix_norm(user_matrix, random_matrix)
        infinity_norm_distances.append((i, inf_norm))
        
        # for every matrix we calculate difference of user matrix and random matrix and store the norms of those differences thats it nothing speciall
        

    
    # sort respect to second component in the list cuz list is like that
    # [(6, np.float64(15.427248620541512)), (0, np.float64(17.378147196982766).......and so on like thatt]
    # first component is just number of the random matrix, second is the actual value of norm of (user matrix -random matrix)
    
    frobenius_distances.sort(key=lambda x: x[1])
    first_norm_distances.sort(key=lambda x: x[1])
    infinity_norm_distances.sort(key=lambda x: x[1])

    
    print("\nThose are the resultss based on the frobenius norm:") # here we use [0][0] + 1 cuz enumerationg starts from zero nothing else
    print(f"Closest matrix is Matrix {frobenius_distances[0][0] + 1} with distance {frobenius_distances[0][1]}") 
    print(f"Farthest matrix is Matrix {frobenius_distances[-1][0] + 1} with distance {frobenius_distances[-1][1]}")
    
    print("\nThose are the resultss based on the first (1) norm:") # max coll sum
    print(f"Closest matrix is Matrix {first_norm_distances[0][0] + 1} with distance {first_norm_distances[0][1]}")
    print(f"Farthest matrix is Matrix {first_norm_distances[-1][0] + 1} with distance {first_norm_distances[-1][1]}")
    
    print("\nthosee are the results based on the infinity norm:")
    print(f"Closest matrix is Matrix {infinity_norm_distances[0][0] + 1} with distance {infinity_norm_distances[0][1]}")
    print(f"Farthest matrix is Matrix {infinity_norm_distances[-1][0] + 1} with distance {infinity_norm_distances[-1][1]}")
    
    
    

if __name__ == "__main__":
    main()


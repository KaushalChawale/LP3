def is_safe(board, row, col):
    # Check this column on upper side
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens(board, row):
    if row == len(board):
        print_board(board)  # Print the solution when a valid configuration is found
        return True  # If you want all solutions, change this logic to not stop here

    for col in range(len(board)):
        if is_safe(board, row, col):
            board[row][col] = 1  # Place the queen
            if solve_n_queens(board, row + 1):  # Recur to place the rest of the queens
                return True
            board[row][col] = 0  # Backtrack and remove the queen

    return False

def print_board(board):
    for row in board:
        print(" ".join('Q' if x else '.' for x in row))
    print()

if __name__ == "__main__":
    N = int(input("Enter the size of the board: "))
    board = [[0 for _ in range(N)] for _ in range(N)]

    solve_n_queens(board, 0)  # Start solving from row 0
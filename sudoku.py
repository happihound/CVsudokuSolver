import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import easyocr
reader = easyocr.Reader(['en'])


def make_Board(image):
    # Find the corners of the sudoku board
    board_corners = allow_user_to_select_corners(image)
    # Find the sudoku board
    board = find_Board(image, board_corners)
    # Resize image
    board = cv2.resize(board, (board.shape[1]*6, board.shape[0]*6), interpolation=cv2.INTER_LANCZOS4)
    # Flip the image mirrow
    board = cv2.flip(board, 1)
    # Display image for debugging
    cv2.imshow('Board', board)
    cv2.waitKey(0)
    # Find the cells of the sudoku board
    cells = cut_board_into_cells(board)
    # OCR the cells
    returnBoard = np.zeros((9, 9))
    for y, row in enumerate(tqdm(cells)):
        for x, cell in enumerate(row):
            result = ocr_cell(cell)
            if len(result) > 0:
                # Uncomment to show every cell with a detected number
                # cv2.imshow('Cell', cell)
                # cv2.waitKey(0)
                returnBoard[y][x] = int(result[0])
    return returnBoard


def allow_user_to_select_corners(image):
    # Display the image
    plt.imshow(image, cmap="gray")
    # Allow the user to select the corners
    corners = plt.ginput(4, timeout=0)
    plt.show()
    # Convert the corners to a numpy array
    corners = np.array(corners, dtype=np.float32)
    # Return the corners
    return corners


def find_Board(image, board_corners):
    # Find the width of the sudoku board
    width = max(np.linalg.norm(board_corners[0] - board_corners[1]),
                np.linalg.norm(board_corners[2] - board_corners[3]))
    # Find the height of the sudoku board
    height = max(np.linalg.norm(board_corners[0] - board_corners[3]),
                 np.linalg.norm(board_corners[1] - board_corners[2]))
    # Find the dimensions of the sudoku board
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    # Find the perspective transform
    perspective_transform = cv2.getPerspectiveTransform(board_corners, dimensions)
    # Find the sudoku board
    board = cv2.warpPerspective(image, perspective_transform, (int(width), int(height)))
    return board


def cut_board_into_cells(board):
    # Find the number of rows and columns
    rows, columns = board.shape
    # Find the number of cells in each row
    cell_rows = rows // 9
    # Find the number of cells in each column
    cell_columns = columns // 9
    # Create a list of cells
    cells = [[0 for i in range(9)] for j in range(9)]
    # Find the cells
    for y in range(9):
        for x in range(9):
            cells[y][x] = board[y * cell_rows: (y + 1) * cell_rows, x * cell_columns: (x + 1) * cell_columns]

    # Return the cells
    return cells


def ocr_cell(image):
    # Blur along 15% of the perimeter of the image
    # This is done to remove the grid lines and any other noise
    # Blur the top and bottom 15% of the image
    image[0:int(image.shape[0]*0.15), :] = cv2.blur(image[0:int(image.shape[0]*0.15), :], (9, 9))
    # Blur the left and right 15% of the image
    image[:, 0:int(image.shape[1]*0.15)] = cv2.blur(image[:, 0:int(image.shape[1]*0.15)], (9, 9))
    # Read the cell
    result = reader.readtext(image, detail=0, paragraph=False, allowlist='123456789')
    # Return the result
    return result


def load_image():
    # Open webcam
    cap = cv2.VideoCapture(0)
    # Read a new frame from webcam
    _, frame = cap.read()
    # Save frame as JPEG file
    # Uncomment this line to use webcam, otherwise it will use the default frame.jpg image provided
    # cv2.imwrite("frame.jpg", frame)
    # Close webcam
    cap.release()
    # Load image
    image = cv2.imread("frame.jpg", cv2.IMREAD_GRAYSCALE)
    # Return image
    return image


def display_board(board):
    # Display the board
    print("\n")
    for x, row in enumerate(board):
        if x == 3 or x == 6:
            print("---------------------")
        for y, cell in enumerate(row):
            print(int(cell), end=" ")
            if y == 2 or y == 5:
                print("|", end=" ")
        print()


# Use backtracking to solve the board
def solve_board(board):
    # Find the next empty cell
    next_empty_cell = find_next_empty_cell(board)
    # Check if the board is solved
    if next_empty_cell is None:
        return board
    else:
        row, column = next_empty_cell
    # Try all numbers
    for number in range(1, 10):
        # Check if the number is valid
        if is_valid_number(board, number, (row, column)):
            # Set the number
            board[row][column] = number
            # Solve the board
            solution = solve_board(board)
            # Check if the board is solved
            if solution is not None:
                return solution
    # Reset the cell
    board[row][column] = 0
    return None


def find_next_empty_cell(board):
    # Find the next empty cell
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell == 0:
                return y, x
    return None


def is_valid_number(board, number, position):
    # Check the row
    for i in range(9):
        if board[position[0]][i] == number and position[1] != i:
            return False
    # Check the column
    for i in range(9):
        if board[i][position[1]] == number and position[0] != i:
            return False
    # Check the box
    box_x = position[1] // 3
    box_y = position[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == number and (i, j) != position:
                return False

    return True


def create_solution_image(board):
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell == 0:
                board[y][x] = 9
    # Create a new image
    new_image = cv2.imread("blank.jpg", cv2.IMREAD_GRAYSCALE)
    new_image = cv2.resize(new_image, (990, 990))
    # Find the number of rows and columns
    rows, columns = new_image.shape
    # Find the number of cells in each row
    cell_rows = rows // 9
    # Find the number of cells in each column
    cell_columns = columns // 9
    # Create a list of cells
    cells = [[0 for i in range(9)] for j in range(9)]
    # Find the cells
    for y in range(9):
        for x in range(9):
            cells[y][x] = new_image[y * cell_rows: (y + 1) * cell_rows, x * cell_columns: (x + 1) * cell_columns]
    # Write the solution on the image
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            cv2.putText(cells[y][x], str(int(cell)), (int(cell_columns * 0.4),
                        int(cell_rows * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    # Create the solution image
    solution_image = np.zeros((rows, columns), dtype=np.uint8)
    for y, row in enumerate(cells):
        for x, cell in enumerate(row):
            solution_image[y * cell_rows: (y + 1) * cell_rows, x * cell_columns: (x + 1) * cell_columns] = cell
    # Return the solution image
    return solution_image


def main():
    # Load image and transform it
    image = load_image()
    # Find the sudoku board
    board = make_Board(image)
    display_board(board)
    # Solve the sudoku board
    board = solve_board(board)
    # Display the solution
    print("Solution:")
    display_board(board)
    # Create the solution image from the solution board
    solution_image = create_solution_image(board)
    # Display the solution image
    cv2.imshow("Solution", solution_image)
    # Wait for the user to press a key
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
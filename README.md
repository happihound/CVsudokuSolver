# CVsudokuSolver
A python script to solve a sudoku board from an image.  
Brief Overview:
1. Take a picture from the webcam
2. Ask user to select the corners of the board, counter clockwise starting at the upper right corner
3. Perform OCR on the image to extract the numbers
4. Solve the sudoku board using backtracking 
5. Display the as an image on a blank sudoku board

It requires the following packages:
```
numpy
cv2
matplotlib
tqdm
easyocr
```
The file is heavily commented for ease of understanding.
Enjoy!

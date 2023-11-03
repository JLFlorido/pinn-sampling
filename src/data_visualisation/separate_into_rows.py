fname = 'results/raw/REP_errors2_gt_none_D3_Hammersley_k1.0c1.0_N2000_L100_error_curves.txt'
# Read the contents of the .txt file
with open(fname, 'r') as file:
    data = file.read()

# Split the data into rows based on the comma
rows = data.split(',')

# Initialize a 2D list to store the rows and columns
matrix = []

# Split each row into columns using spaces as separators
for row in rows:
    columns = row.split()
    # Convert the column values to integers (assuming they are integers)
    int_columns = [column for column in columns]
    matrix.append(int_columns)

# Now, 'matrix' contains the data in rows and columns format
# You can access specific elements like matrix[row_index][column_index]


# Open the output file for writing

with open(fname, 'w') as file:
    for row in matrix:
        row_str = ' '.join(map(str, row))
        file.write(row_str + '\n')

print(f"Matrix saved to {fname}")
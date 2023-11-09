#!/bin/bash

# Check for required argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Assign arguments to variables
SOURCE_DIR=$1
DEST_DIR=$2

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Function to comment out numba dependencies
comment_numba_deps() {
    local file=$1
    local dest_file=$2
    sed '/from numba import/ s/^/#/' "$file" | sed '/import numba/ s/^/#/' | sed '/@jit/ s/^/#/' | sed '/jit(/ s/^/#/' > "$dest_file"
    #sed '/from numba import/ s/^/#/' "$file" | sed '/@jit/ s/^/#/' | sed '/jit(/ s/^/#/' > "$new_file"
}

# Export the function to be available in find's exec command
export -f comment_numba_deps

# Find all Python files and apply the transformation
find "$SOURCE_DIR" -name '*.py' -exec bash -c 'comment_numba_deps "$0" "'"$DEST_DIR"'/$(basename "$0")"' {} \;

echo "Transformation complete. Python files with numba dependencies commented are in $DEST_DIR"

#for file in $(find $SOURCE_DIR -type f -exec basename {} \;); do diff $SOURCE_DIR/$file $DEST_DIR /$file; done

zip -r "${DEST_DIR}.zip" "$DEST_DIR"

echo "The transformed Python files are zipped in ${DEST_DIR}.zip"


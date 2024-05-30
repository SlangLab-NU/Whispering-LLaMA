#!/bin/bash

# Define the maximum file size you want to track
MAXSIZE=100M

# Create or overwrite the .gitignore file
echo "# .gitignore" > .gitignore

# Find files larger than MAXSIZE and add them to .gitignore
find . -size +$MAXSIZE -type f | sed 's|^\./||' >> .gitignore
#!/bin/bash

# Extract image file paths from README.md
images=$(grep -oE '!\[.*\]\(([^)]+)\)' README.md | sed -E 's/!\[.*\]\(([^)]+)\)/\1/')

# Add images to Git if they exist
for img in $images; do
    if [[ -f "$img" ]]; then
        git add -f "$img"
        echo "Added: $img"
    else
        echo "Warning: Image not found - $img"
    fi
done

# Commit changes
git commit -m "Added images referenced in README"
echo "All referenced images added to Git."

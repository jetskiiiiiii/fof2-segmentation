#!/bin/bash
for file in *\_*.png; do
    # Strip everything from the first underscore to the end 
	newname="${file%%png*.png}.png" 
	mv "$file" "$newname"
done

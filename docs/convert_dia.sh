#!/bin/bash
STATICDIR="source/_static"
DIAFILES="$STATICDIR/*.dia"

if type dia &> /dev/null; then
    for file in $DIAFILES
    do
        dia -e ${file%.dia}.png $file
    done
else
    echo "Dia is not installed, cannot convert $DIAFILES to png!"
fi

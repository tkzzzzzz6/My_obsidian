#!/bin/bash

m = `echo $RANDOM`

n1 = $[$m % 100]

while
do 
    rea -p "Please input a number:"n
    if [ $n == $n1]
    then
        break
elif [ $n-gt$n1]
then
echo "bigger"
continue
else
echo "smaller"
continue
fi
done
echo "You are right"
    
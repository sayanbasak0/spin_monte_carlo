# #!/bin/bash -l
# # submit.job
# # insert job requirements ...
# ./a.out; chk=$?

# if [[ chk –eq 2 ]]; then
#   qsub submit.job
# fi

gcc -fopenmp temp.c
./a.out
chk=$?

if [[ chk -eq $1 ]]; then
	echo $chk
else
	echo $chk "not 0"
fi


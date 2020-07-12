for dim in 128 256 512
do
    echo "===== $dim x $dim"
    for nt in 1 2 4 8 12
    do
        out=out_$nt
        OMP_NUM_THREADS=$nt srun -n1 -c12 --hint=nomultithread ./main $dim $dim 200 0.01 | grep 'rate of' > $out
        rate=`awk '{printf("%10.1f", $8)}' $out `
        printf "%8.1f " $rate
    done
    printf "\n"
done

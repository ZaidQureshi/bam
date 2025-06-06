for n in {1..15}; do
	nq=$(( n * 32 ))
	echo "num_queues=${nq}"
	sudo ./build/bin/nvm-block-bench --ssd 2 -g 0 --page_size=4096 --threads=$(( 8 << 20 )) --pages=$(( 8 << 20 )) --num_blks=$(( 1 << (40-12) )) --random true --reqs=32 --num_queues=${nq} |& grep Effective
done


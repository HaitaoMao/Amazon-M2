for lr in 0.01 0.001 0.0001
do
python session_example.py --dataset=amazon --gpu=0 --model=STAMP --lr=$lr --debug 1 2>&1 | tee -a ./result/STAMP_${lr}_${dropout}_${num_layers}.log;
done
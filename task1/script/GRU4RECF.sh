for lr in 0.01 0.001 0.0001
do
for num_layers in 1 2 3
do
python session_example.py --dataset=amazon --gpu=0 --model=GRU4RecF --lr=$lr --num_layers=$num_layers --debug 1 2>&1 | tee -a ./result/GRU4RecF_${lr}_${num_layers}.log;
done
done

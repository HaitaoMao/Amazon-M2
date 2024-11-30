for lr in 0.01 0.001 0.0001
do
for step in 1 2
do
python session_example.py --dataset=amazon --gpu=0 --model=SRGNN --lr=$lr --step=$step --debug 1 2>&1 | tee -a ./result/SRGNN_${lr}_${dropout}_${num_layers}.log;
done
done
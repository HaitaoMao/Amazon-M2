for lr in 0.01 0.001 0.0001
do
for n_layers in 1 2
do
for dropouts in '[0.25,0.5]' '[0.2,0.2]' '[0.1,0.2]'
do
python session_example.py --dataset=amazon --gpu=0 --model=NARM --lr=$lr --n_layers=$n_layers --dropouts=$dropouts --debug 1 2>&1 | tee -a ./result/NARM_${lr}_${dropout}_${num_layers}.log;
done
done
done
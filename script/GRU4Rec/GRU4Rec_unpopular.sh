for lr in 0.01 0.001 0.0001
do
for dropout in 0.0 0.1 0.2 0.3 0.4 0.5
do
for num_layers in 1 2 3
do
python train.py --dataset=amazon_unpopular --gpu=2 --model=GRU4Rec --lr=$lr --dropout=$dropout --num_layers=$num_layers
done
done
done
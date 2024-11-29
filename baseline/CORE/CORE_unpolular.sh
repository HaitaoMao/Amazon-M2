for lr in 0.001 0.0001
do
for dropout in 0.2 0.5
do
for num_layers in 1 2
do
python train.py --dataset=amazon_unpopular --gpu=0 --model=CORE --lr=$lr --hidden_dropout=$dropout --attn_dropout=$dropout --num_layers=$num_layers
done
done
done

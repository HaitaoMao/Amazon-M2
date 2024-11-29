for lr in 0.01 0.001 0.0001
do
for dropout in 0.0 0.1 0.2 0.3 0.4 0.5
do
for num_layers in 1 2 3
do
python finetune.py --dataset=amazon --gpu=1 --model=GRU4Rec --lr=$lr --dropout=$dropout --num_layers=$num_layers --saved_model='GRU4Rec-Jun-04-2023_12-04-08.pth'
done
done
done
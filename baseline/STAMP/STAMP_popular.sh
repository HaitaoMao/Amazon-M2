for lr in 0.001 0.0001
do
python train.py --dataset=amazon_popular --gpu=2 --model=STAMP --lr=$lr
done
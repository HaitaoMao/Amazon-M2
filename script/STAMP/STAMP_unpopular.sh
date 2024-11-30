for lr in 0.01 0.001 0.0001
do
python train.py --dataset=amazon_unpopular --gpu=3 --model=STAMP --lr=$lr
done
for lr in 0.01 0.001 0.0001
do
for step in 1 2
do
python train.py --dataset=amazon_unpopular --gpu=5 --model=SRGNN --lr=$lr --step=$step
done
done
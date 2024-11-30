for lr in 0.01 0.001 0.0001
do
for n_layers in 1 2
do
for dropouts in '[0.25,0.5]' '[0.2,0.2]' '[0.1,0.2]'
do
python train.py --dataset=amazon_unpopular --gpu=2 --model=NARM --lr=$lr --n_layers=$n_layers --dropouts=$dropouts
done
done
done
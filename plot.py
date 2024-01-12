import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from collections import Counter

def plot_frequency(xs, ys, dataset):
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(xs)):
        if ys[i] < 3:
            xs = xs[:i]
            ys = ys[:i]
            break

    plt.plot(xs, ys, color="black", linewidth=4)
    ax.grid(False) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    plt.yscale('log')
    plt.ylabel('#Products', fontsize=38, fontfamily='serif')
    plt.xlabel("Products frequency", fontsize=38,  fontfamily='serif') 
    
    plt.xticks(fontsize=32, fontfamily='serif')
    plt.yticks(fontsize=32, fontfamily='serif')
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig(f"fig/frequency/{dataset}.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"fig/frequency/{dataset}.pdf", dpi=1000, bbox_inches='tight')

def dist_plot_single(datas, dataset):
    counter = Counter()
    for data in datas: 
        counter.update([data])
    xs, ys = np.array(list(counter.keys())), np.array(list(counter.values()))
    index = np.argsort(xs)
    xs, ys = xs[index], ys[index]
    fig, ax1 = plt.subplots(figsize=(8,8))
    plt.yscale('log')
    plt.plot(xs, ys, color="black", linewidth=3)
      
    ax1.grid(False) 
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.ylabel('#Session', fontsize=38, fontfamily='serif') 
    plt.xlabel("Session length", fontsize=38,  fontfamily='serif') 
    plt.xticks(fontsize=32, fontfamily='serif')
    plt.yticks(fontsize=32, fontfamily='serif')
    plt.savefig(f'fig/length/{dataset}.png', bbox_inches='tight' ) # 
    plt.savefig(f'fig/length/{dataset}.pdf', bbox_inches='tight' ) # 

    plt.close()


def dist_plot_repeat(datas, dataset):
    counter = Counter()
    for data in datas: 
        counter.update([data])
    xs, ys = np.array(list(counter.keys())), np.array(list(counter.values()))
    index = np.argsort(xs)
    xs, ys = xs[index], ys[index]
    fig, ax1 = plt.subplots(figsize=(8,8))
    plt.yscale('log')
    plt.plot(xs, ys, color="black", linewidth=3)
    
    ax1.grid(False)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
    plt.ylabel('#Session', fontsize=38, fontfamily='serif') 
    plt.xlabel("#Repeated products", fontsize=38,  fontfamily='serif') 
    plt.xticks(fontsize=32, fontfamily='serif')
    plt.yticks(fontsize=32, fontfamily='serif')
    plt.savefig(f'fig/repeat/{dataset}.png', bbox_inches='tight' ) 
    plt.savefig(f'fig/repeat/{dataset}.pdf', bbox_inches='tight' ) 

    plt.close()


def dist_plot_coll(datas, dataset):
    fig, ax1 = plt.subplots(figsize=(8,8))
    counter = Counter()
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax1 = plt.subplots(figsize=(8,8))
    sns.kdeplot(datas, fill=True, color=colors[0],cbar_ax=ax1, linewidth=2, alpha=0.3)
            
    plt.xlim(0, 2.5) 
    ys = [round(i * 0.2, 1) for i in range(6)]
    ystrs = [str(round(y, 1)) for y in ys]
    
    xs = [round(i * 0.5, 1) for i in range(1, 6)]
    xstrs = [str(round(x, 1)) for x in xs]
    ax1.grid(False)
    
    plt.ylabel('Density', fontsize=38, fontfamily='serif') 
    plt.xlabel( "SKNN score", fontsize=38,  fontfamily='serif') 
    plt.xticks(xs, xstrs,fontsize=32, fontfamily='serif')
    plt.yticks(ys, ystrs,fontsize=32, fontfamily='serif')
    plt.savefig(f'fig/coll/{dataset}.png', bbox_inches='tight' )  
    plt.savefig(f'fig/coll/{dataset}.pdf', bbox_inches='tight' )  

    plt.close()


def repeat_bar(datas):
    fig, ax = plt.subplots(figsize=(8,8))
    keys = list(datas.keys())
    values = list(datas.values())
    num_group = 1
    num_model = len(datas)
    bar_width = 0.8 / num_group
    
    xs = bar_width * np.arange(num_group)
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    for idx, (key, value) in enumerate(zip(keys, values)):
        ax.bar(xs + idx * bar_width, value, bar_width, color=colors[idx], label=key,capsize=2,edgecolor='black')
    xs = [xs[0] + idx * bar_width for idx in range(len(keys))]
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) 
    plt.ylabel('Repeat proportion', fontsize=34, fontfamily='serif') 
    plt.xlabel('Locales', fontsize=34, fontfamily='serif') 
    plt.yticks(fontsize=28, fontfamily='serif')
    
    ax.set_xticks(xs, list(keys), fontsize=28, fontfamily='serif')
    
    plt.savefig(f"fig/repeat/repeat_ratio.png", bbox_inches='tight')
    plt.savefig(f"fig/repeat/repeat_ratio.pdf", bbox_inches='tight')



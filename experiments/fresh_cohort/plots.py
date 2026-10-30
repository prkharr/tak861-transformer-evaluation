"""Aggregate diagnostics from saved training histories and locked predictions."""
import numpy as np
from matplotlib.figure import Figure


def training_curves(results):
    arms = list(results)
    fig = Figure(figsize=(12, 3.5*len(arms)), layout='constrained')
    axes = fig.subplots(len(arms),2,squeeze=False)
    for i,arm in enumerate(arms):
        for j,run in enumerate(results[arm]):
            h = run['history']
            seed = run['identity']['seed']
            epochs = [r['epoch'] for r in h]
            color = 'C'+str(j%10)
            for ax,train,val,title in ((axes[i,0],'train_lift','validation_lift','Lift at top 10%'),
                                      (axes[i,1],'train_log_loss','validation_log_loss','Log loss')):
                ax.plot(epochs,[r[train] for r in h],color=color,linestyle='--',alpha=.7,label=str(seed)+' train')
                ax.plot(epochs,[r[val] for r in h],color=color,label=str(seed)+' validation')
                ax.axvline(run['summary']['best_epoch'],color=color,alpha=.25,linewidth=1)
                ax.set(title=arm+' — '+title,xlabel='Epoch',ylabel=title)
                ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Training versus validation; vertical lines mark selected epochs. No test-based tuning.')
    return fig


def split_lifts(lock, test_report=None):
    arms = list(lock['metrics'])
    groups = [('train','Training (resubstitution)'),('validation','Validation (selected)')]
    if test_report is not None:
        if test_report['split'] != 'test' or not test_report.get('test_prediction_performed'):
            raise ValueError('Test chart requires actual locked test predictions')
        groups.append(('test','Test (retrospective)'))
    fig = Figure(figsize=(max(8,len(arms)*2),5),layout='constrained')
    ax = fig.subplots()
    width=.8/len(groups); x=np.arange(len(arms))
    for i,(split,label) in enumerate(groups):
        values=[test_report['metrics'][a]['primary_lift'] if split=='test'
                else lock['metrics'][a][split]['primary_lift'] for a in arms]
        bars=ax.bar(x-.4+width*(i+.5),values,width,label=label)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
    ax.axhline(1,color='gray',linestyle=':',label='Random ranking')
    ax.set(xticks=x,xticklabels=arms,ylabel='Lift at top 10%',title='Generalization diagnostic on frozen splits')
    ax.tick_params(axis='x',labelrotation=15); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    return fig


def lift_ranking(table, scope):
    required={'Model','Lift@10%'}
    if not required <= set(table):
        raise ValueError('Expected model names and computed lift values')
    fig = Figure(figsize=(9,max(4,.55*len(table))),layout='constrained')
    ax=fig.subplots()
    ordered=table.sort_values('Lift@10%',ascending=True)
    bars=ax.barh(ordered.Model,ordered['Lift@10%'])
    ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.axvline(1,color='gray',linestyle=':')
    ax.set(xlabel='Lift at top 10%',title=scope)
    ax.grid(axis='x',alpha=.2)
    return fig

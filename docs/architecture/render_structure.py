"""Render the implemented model and the separate feature-selection branch."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(20, 14))
fig.patch.set_facecolor('#f4f7fb')
ax.set(xlim=(0, 20), ylim=(0, 14))
ax.axis('off')
ax.text(.5, 13.5, 'TAK861 | Transformer structure', fontsize=26, weight='bold', color='#142b49')
ax.text(.5, 13.08, 'Implemented snapshot-padding pipeline and feature-selection extension • 30 September 2026', fontsize=12, color='#50627a')

def box(x, y, w, h, title, body, color='#ffffff', edge='#b8c8dc'):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.12,rounding_size=0.12', facecolor=color, edgecolor=edge, linewidth=1.3))
    ax.text(x+.18,y+h-.25,title,va='top',fontsize=13,weight='bold',color='#142b49')
    ax.text(x+.18,y+h-.68,body,va='top',fontsize=10.5,linespacing=1.5,color='#233b56')

def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#54718e',lw=1.7))

ax.text(.5,12.55,'01  INPUTS & PREPARATION',fontsize=12,weight='bold',color='#227c87')
ax.text(7.1,12.55,'02  MODEL FOR EACH REPRESENTATION',fontsize=12,weight='bold',color='#227c87')
ax.text(13.7,12.55,'03  TRAINING & EVALUATION',fontsize=12,weight='bold',color='#227c87')

box(.5,10.25,5.8,2.0,'Frozen V63 population + 49 features',
    '23,151 snapshots • 12,447 patients\nMODEL_DATA joined by patient + snapshot date\nRESP is the target; identifiers are excluded\nExact keys, labels and feature order checked')
box(.5,7.6,5.8,2.25,'Calendar grid + enrollment validity',
    'Monthly: 12 steps | Quarterly: 4 steps\nStep 0 = most recent period\nAny enrollment overlap makes a period valid\nRepeat snapshot values at valid steps only\nUnavailable periods = zero + explicit mask')
box(.5,4.95,5.8,2.25,'Patient split + frozen preprocessing',
    'Same patient stays in one split\nTRAIN / VALIDATION / TEST remain fixed\nShared TRAIN-only imputation and scaling\nFit once per eligible TRAIN snapshot\nKeep padded values zero after transformation')
box(.5,2.8,5.8,1.75,'Two independent model inputs',
    'Monthly: [batch, 12, 49]\nQuarterly: [batch, 4, 49]\nValidity mask: [batch, steps]')
for upper,lower in [(10.25,9.85),(7.6,7.2),(4.95,4.55)]: arrow(3.4,upper,3.4,lower)

box(7.1,10.65,5.8,1.6,'Feature projection + temporal positions',
    'Linear: 49 → 128 dimensions\nAdd learned position vector for each step\nOutput: [batch, steps, 128]',color='#eaf4fa')
box(7.1,7.6,5.8,2.65,'Transformer encoder × 2',
    'Pre-LayerNorm → 4-head self-attention\nDropout + residual connection\nPre-LayerNorm → feedforward 128 → 256 → 128\nReLU + dropout + residual connection\nDropout = 0.20; key padding mask = NOT valid\nAttention operates across timesteps',color='#eaf4fa')
box(7.1,5.3,5.8,1.9,'Final LayerNorm + masked mean',
    'Exclude padded outputs from the average\nPooled representation: [batch, 128]\nAll-padded rows bypass encoder and use\na zero pooled vector through the same head',color='#eaf4fa')
box(7.1,2.8,5.8,2.1,'Prediction head',
    'Linear 128 → 64 → ReLU → Dropout 0.20\nLinear 64 → 1 logit\nSigmoid → response ranking score\nCalibration has not been established',color='#eaf4fa')
for upper,lower in [(10.65,10.25),(7.6,7.2),(5.3,4.9)]: arrow(10,upper,10,lower)
ax.plot([6.43,6.7,6.7],[3.7,3.7,11.45],color='#54718e',lw=1.5)
arrow(6.7,11.45,6.98,11.45)

box(13.7,9.9,5.8,2.35,'Optimization defaults',
    'TRAIN-weighted binary cross-entropy\nExplicit L1 = 1e-6; explicit L2 = 1e-4\nPenalties on linear and attention weights\nAdamW: learning rate 0.001; weight decay 0\nBatch 64 • gradient clipping 1.0\nMaximum 20 epochs • patience 5')
box(13.7,7.4,5.8,2.1,'Checkpoint selection',
    'Early stopping on VALIDATION average precision\nRestore highest validation-AP checkpoint\nChoose threshold on VALIDATION\nSave model, preprocessing and input fingerprints\nMonthly and quarterly trained separately')
box(13.7,4.8,5.8,2.2,'Frozen evaluation + interpretation',
    'Average precision, ROC-AUC, precision / recall\nTop-10% lift, capture and decile curves\nTrain–validation gap and learning curves\nPermutation importance + feature correlations\nTEST is reserved for frozen model assessment')
ax.plot([13.03,13.3,13.3],[3.7,3.7,11.45],color='#54718e',lw=1.5)
arrow(13.3,11.45,13.58,11.45)
arrow(16.6,9.9,16.6,9.5)
arrow(16.6,7.4,16.6,7.0)
box(13.7,2.8,5.8,1.6,'Interpretation boundary',
    'Repeated snapshot inputs are not history.\nQuarterly values are not monthly aggregates\nin this 49-feature snapshot-padding route.',color='#fff2db',edge='#d8ad59')

box(.5,.4,19,1.85,'PARALLEL EXTENSION | Feature selection from 1,028 claims categories',
    'Saved monthly counts → 12-month totals → latest TRAIN snapshot per patient → quality checks → three TRAIN-fold selectors\n'
    'FILTER: mutual information + redundancy pruning (100)   |   WRAPPER: shortlist + RFE (50)   |   EMBEDDED: elastic net + trees (100)\n'
    'Compared with a fixed L2-logistic evaluator: AP ≈ 0.193 / 0.195 / 0.193 versus 0.156 for all eligible features.\n'
    'These are feature-selection results, not Transformer results. Selected-feature Transformer comparison is the next experiment.',color='#e9f4ee',edge='#82ad95')

for ext in ('png','svg','pdf'):
    fig.savefig(OUT / ('transformer_structure.'+ext),dpi=170,bbox_inches='tight',facecolor=fig.get_facecolor())
plt.close(fig)

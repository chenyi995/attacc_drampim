"""Local copy of the paper's shared style; no repository/data imports."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kvchime-reproduce-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
ROOT = Path(__file__).resolve().parent
_template = json.loads((ROOT / 'template.json').read_text())
SPECS = {_template['label']: _template}
SINGLE, DOUBLE = 3.33395599833956, 7.0
LEGACY_FONT = dict(title=8.5, label=8.0, legend=7.0, note=6.5)
FONT = dict(title=8.5, label=8.0, legend=7.0, note=6.5)
COLOR = dict(base='#C9825B', improved='#377E9B', shared='#DDEBF1', private='#F2DFD2', state='#E6E8EB', ink='#333333')
HEATMAP_COLORS=['#A8182B','#FFFFFF','#1856A0']

def configure(fonts=None):
    fonts=FONT if fonts is None else fonts
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':fonts['note'],
        'axes.titlesize':fonts['title'],'axes.labelsize':fonts['label'],
        'legend.fontsize':fonts.get('legend',fonts['label']),'xtick.labelsize':fonts['note'],'ytick.labelsize':fonts['note'],
        'axes.linewidth':.6,'lines.linewidth':1.25,'patch.linewidth':.5,
        'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':True,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':None})

def canvas(label, **kw):
    s=SPECS[label]
    configure(s['fonts_pt'])
    return plt.subplots(figsize=(s['width_inches'],s['height_inches']),**kw)

def save(fig, label, path):
    fig.canvas.draw()
    sizes=set()
    texts=[]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            value=t.get_text()
            size=t.get_fontsize()
            assert size in SPECS[label]['fonts_pt'].values(), (label,value,size)
            sizes.add(size);texts.append(dict(text=value,font_pt=size))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,metadata={'CreationDate':None,'ModDate':None,'Creator':'KVChime shared figure templates'})
    fig.savefig(path.with_suffix('.png'),dpi=180)
    plt.close(fig)
    return dict(label=label,template=f'fig/plots/templates/{label}.json',width_inches=SPECS[label]['width_inches'],
                height_inches=SPECS[label]['height_inches'],font_sizes_pt=sorted(sizes),text=texts,path=str(path))

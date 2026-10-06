"""Export the current Markdown report, including its tables, as one PDF page."""
from pathlib import Path
import re
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mariners-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pypdf import PdfReader, PdfWriter, Transformation

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'exploration/game_00000000_part2.md'
text=SOURCE.read_text()
# Render only the chart portion of the plotting script, preserving edited prose.
import sys
sys.path.insert(0,str(HERE/'exploration'))
plot_script=HERE/'exploration/plot_game_close_pitches.py'
plot_code=plot_script.read_text().split('\nrows=[]',1)[0]
plot_code=plot_code.replace("fig.savefig(here/name,dpi=180,bbox_inches='tight');", "fig.savefig((here/name).with_suffix('.pdf'),bbox_inches='tight');")
exec(compile(plot_code,str(plot_script),'exec'),{'__file__':str(plot_script)})
vector_charts=[]
preview_images=[]
fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
title=re.search(r'^# (.+)$',text,re.M).group(1)
fig.text(.065,.967,title,fontsize=17,fontweight='bold',va='top')
intro=text.split('## ',1)[0].split('\n',1)[1].strip().replace('**','')
fig.text(.065,.935,intro,fontsize=9,va='top')
for i,section in enumerate(text.split('\n## ')[1:]):
    heading=section.splitlines()[0]
    top=.896 if i==0 else .490
    fig.text(.065,top,heading,fontsize=11,fontweight='bold',va='top')
    image_name=re.search(r'!\[.*?\]\((.*?)\)',section).group(1)
    image=mpimg.imread(SOURCE.parent/image_name)
    ax=fig.add_axes([.065,top-.215,.87,.190]);preview_images.append(ax.imshow(image));ax.axis('off')
    vector_charts.append((SOURCE.parent/Path(image_name).with_suffix('.pdf'),top))
    lines=[line for line in section.splitlines() if line.startswith('|')]
    headers=[v.strip() for v in lines[0].strip('|').split('|')]
    headers=[v.replace('Overturn likelihood','Overturn\nlikelihood').replace('Pitch cluster','Pitch\ncluster').replace('Challenges remaining','Challenges\nremaining').replace('WP change if correct','WP change\nif correct').replace('Catcher movement','Catcher\nmovement') for v in headers]
    rows=[[v.strip() for v in line.strip('|').split('|')] for line in lines[2:]]
    height=.021*(len(rows)+1)
    tax=fig.add_axes([.065,top-.235-height,.87,height]);tax.axis('off')
    widths=[.06,.105,.115,.07,.095,.12,.05,.135,.13,.12]
    table=tax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(7)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d7dee5');cell.set_linewidth(.5)
        cell.set_facecolor('#e8eef4' if r==0 else ('#f7f9fb' if r%2 else 'white'))
        if r==0:cell.set_text_props(fontweight='bold',fontsize=6.6)
for artist in preview_images:artist.set_visible(False)
fig.savefig(HERE/'game_00000000_report.pdf',format='pdf')
for artist in preview_images:artist.set_visible(True)
fig.savefig(HERE/'game_00000000_report_preview.png',dpi=150)
plt.close(fig)
pdf_path=HERE/'game_00000000_report.pdf'
page=PdfReader(pdf_path).pages[0]
page_width=float(page.mediabox.width);page_height=float(page.mediabox.height)
for chart_path,top in vector_charts:
    chart=PdfReader(chart_path).pages[0]
    cw=float(chart.mediabox.width);ch=float(chart.mediabox.height)
    width=.87*page_width;height=.190*page_height
    scale=min(width/cw,height/ch)
    left=.065*page_width+(width-cw*scale)/2
    bottom=(top-.215)*page_height+(height-ch*scale)/2
    page.merge_transformed_page(chart,Transformation().scale(scale).translate(left,bottom))
writer=PdfWriter();writer.add_page(page)
with pdf_path.open('wb') as output:writer.write(output)
assert len(list(PdfReader(pdf_path).pages[0].images))==0, 'Raster images found in vector PDF'
assert len(PdfReader(HERE/'game_00000000_report.pdf').pages)==1
print(HERE/'game_00000000_report.pdf')

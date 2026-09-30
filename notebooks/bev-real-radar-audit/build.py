import json
src = open('radar_audit.py', encoding='utf-8').read()
c1 = "import os\nopen('/kaggle/working/radar_audit.py','w',encoding='utf-8').write(" + repr(src) + ")\nprint('written')"
c2 = "%%bash\nset -e\nmkdir -p /kaggle/working/radar_audit\npython3 /kaggle/working/radar_audit.py /kaggle/working/radar_audit 2>&1 | tail -80"
cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)} for c in (c1, c2)]
json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}, open('bev-real-radar-audit.ipynb', 'w'), indent=1)
json.dump({"id": "canhquang/bev-real-radar-audit", "title": "bev_real_radar_audit", "code_file": "bev-real-radar-audit.ipynb", "language": "python", "kernel_type": "notebook",
  "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "dataset_sources": ["aadimator/nuscenes-mini"], "competition_sources": [], "kernel_sources": []}, open('kernel-metadata.json', 'w'), indent=2)

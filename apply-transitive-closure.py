#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re

import numpy as np
from scipy.sparse.csgraph import shortest_path
import pandas as pd

pd.options.display.max_colwidth=500

ent_pattern = re.compile('T(\d+)\t([A-Z_]+) (\d+ \d+(;\d+ \d+)*)\t(.+)')
rel_pattern = re.compile('R(\d+)\t([A-Z_]+) Arg1:T(\d+) Arg2:T(\d+)')
ref_pattern = re.compile('R(\d+)\tCOREFERENCE Arg1:T(\d+) Arg2:T(\d+)')

folder = './test/sample/'

for filename in os.listdir(folder):
    if filename.endswith('.ann'):
        with open(folder + filename, 'r', encoding='utf8') as fin:
            text = fin.read()

        rels = rel_pattern.findall(text)
        if len(rels) == 0:
            continue
        assert len(rels) == int(rels[-1][0])

        df_ent = pd.DataFrame([(e[0], e[1], e[2], e[4]) for e in ent_pattern.findall(text)], 
                  columns=['id', 'type', 'range', 'text']).set_index('id')
        df_ref = pd.DataFrame(ref_pattern.findall(text), columns=['id', 'ent1', 'ent2'])
        df = df_ref.join(df_ent, on='ent1', rsuffix='1').join(df_ent, on='ent2', rsuffix='2')
        
        common_span = set(df['range']) & set(df['range2'])
        if len(common_span) != 0:
            print(filename)
            display(df)

            coref_dict = {r:e for e, r in zip(df['ent1'], df['range'])}
            ent_dict = {r:e for e, r in zip(df['ent2'], df['range2'])}

            all_spans = sorted(set(df['range']) | set(df['range2']))
            num_spans = len(all_spans)

            df_graph = pd.DataFrame(np.zeros((num_spans, num_spans), dtype=int), index=all_spans, columns=all_spans)

            for r1, r2 in zip(df['range'], df['range2']):
                df_graph.loc[r1, r2] = 1

            d = shortest_path(df_graph.values, directed=True)

            print('new rels')
            new_rels = []
            for e1, e2 in zip(*np.where(((~np.isinf(d)) & (d>1)))):
                r1 = all_spans[e1]
                r2 = all_spans[e2]
                new_rels.append((coref_dict[r1], ent_dict[r2]))

            with open(folder + filename, 'a', encoding='utf8') as fout:
                for i, (e1, e2) in enumerate(new_rels, start=len(rels)+1):
                    rel_str = 'R%d\tCOREFERENCE Arg1:T%s Arg2:T%s'%(i, e1, e2)
                    print(rel_str)
                    fout.write(rel_str + '\n')                        


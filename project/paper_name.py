#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:37:19 2020

@author: ebukaokpala
"""

import feedparser
import urllib.request as libreq

def main():
    
    path = "cocitation-results/"
    with open(path + 'cranks_and_cmranks.txt', 'r') as f:
        for line in f:
            id, citation_count, rank, degree_rank, betweenness_rank, closeness_rank, page_rank = line.split()
            padding = 7 - len(id)
            paper_id = '0' * padding + id

            with libreq.urlopen('http://export.arxiv.org/api/query?id_list=' + 'hep-ph/' + paper_id) as url:
                response = url.read()
            feed = feedparser.parse(response)
            title = feed.entries[0].title
            title = title.replace("\n", "")

            with open(path + 'title_of_top_30_papers_cranks_cmranks', 'a') as f:
                f.write(f"{title} & {citation_count} & {rank} & {degree_rank} & {betweenness_rank} & {closeness_rank} & {page_rank}\n")
                    
    
if __name__ == "__main__":
    main()
    print("Done!.")
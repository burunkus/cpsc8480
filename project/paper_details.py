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
    paper_files = ['top_betweenness_cen_cocitation-92-00.txt', 'top_betweenness_cen_cocitation-92-02.txt',
                   'top_betweenness_cen_cocitation-92-94.txt', 'top_betweenness_cen_cocitation-92-96.txt', 
                   'top_betweenness_cen_cocitation-92-98.txt', 'top_closeness_cen_cocitation-92-00.txt', 
                   'top_closeness_cen_cocitation-92-02.txt', 'top_closeness_cen_cocitation-92-94.txt', 
                   'top_closeness_cen_cocitation-92-96.txt', 'top_closeness_cen_cocitation-92-98.txt', 
                   'top_degree_cen_cocitation-92-00.txt', 'top_degree_cen_cocitation-92-02.txt', 
                   'top_degree_cen_cocitation-92-94.txt', 'top_degree_cen_cocitation-92-96.txt',
                   'top_degree_cen_cocitation-92-98.txt', 'top_page_rank_cocitation-92-00.txt',
                   'top_page_rank_cocitation-92-02.txt', 'top_page_rank_cocitation-92-94.txt',
                   'top_page_rank_cocitation-92-96.txt', 'top_page_rank_cocitation-92-98.txt'
                  ]
    
    for file in paper_files:
        with open(path + file, 'r') as f:
            name = file.split('.')[0] + '_details.txt'
            save_path = "cocitation-results/top_centrality_details_whole/"
            for line in f:
                id = line.split()[0]
                padding = 7 - len(id)
                paper_id = '0' * padding + id
                     
                with libreq.urlopen('http://export.arxiv.org/api/query?id_list=' + 'hep-ph/' + paper_id) as url:
                    response = url.read()
                feed = feedparser.parse(response)
                title = feed.entries[0].title
                title = title.replace("\n", "")
                authors_list = feed.entries[0].authors
                authors = []
                for author in authors_list:
                    authors.append(author.name)
                authors = ", ".join(authors)

                with open(save_path + name, 'a') as f:
                    f.write(f"{title} by {authors} \n")
                    

    
if __name__ == "__main__":
    main()
    print("Done!.")


**Thesisaurus: A Virtual Assistant for Curating and Synthesizing Thesis Research**

**CMPE 259 Project Proposal** 

**by**

**Khush Naidu**  
**SJSU ID: 015798328**

In the field of research, the process of organizing thematic narratives across different papers, keeping important details within coherent matrices, recalling precise experimental results, and performing substantial recall can be one of the most time-consuming aspects of the practice. Most researchers still synthesize papers by hand in the first, second, and third passes, but it can be very challenging to manually maintain narratives without compromising the content when the corpus of papers exceeds 20\. For a while now, LLMs have been doing language synthesis tasks. While a number of AI-supported tools have addressed the need to synthesize individual documents for direct recall and Q\&A tasks, I haven't come across any tool that can combine information from different documents and curate the queries for tasks for the research pipeline in particular.   
	As a student currently working on a master’s thesis, a tool like this would be ideal for me to make reading plans, compare experimental results across papers, cite papers accurately, and organize thematic details comprehensively. While this tool could also be used to help write the paper, that is something that would border on the lines of plagiarism, and so I want to curate the virtual assistant to perform very objective tasks that serve as nothing more than helpers for an ethical researcher’s reading and organizing efforts. I will build a VA that operates on a thesis corpus to (a) create reading plans, (b) compare results across papers, (c) surface accurate citations/snippets, and (d) organize themes (methods, datasets, metrics, limitations). The tool is designed for objective organization, not ghostwriting: it summarizes, verifies, and cites and does not perform extensive text generation beyond concise, source-grounded synthesis. With that in mind, here are some generic queries that could be valuable for the VA to answer:

1. Can you compile a list of all the common evaluation datasets in the corpus?  
2. What are the commonly used vision models across all papers?  
3. Do any of the papers identify similar limitations in their outcomes?  
4. Can you cluster papers into a literature matrix based on which part of the thesis pipeline they explore?  
5. Extract a table of training setups (optimizer, LR, batch size, epochs, augmentations, pretrained weights).  
6. Give me section-wise guides for how to most effectively read the papers.  
7. What are the 5 most-cited papers in my corpus on \[subtopic\]?  
8. Compare papers A, B, and C on experimental parity.  
9. Create a reading order: prerequisites, core, and extensions, with a 1-line rationale per paper.  
10. Compile every technology used across papers and arrange them in a task/usage-based matrix.  
11. Summarize the ablation studies reported and the primary takeaways per paper.  
12. Identify evaluation harnesses used across papers and recommend which ones are feasible for my sandboxing efforts.  
13. Compare the benchmarks used to evaluate the experimental results from each paper.  
14. List each paper’s core claims and the exact protocols used to test them.  
15. List all the hardware introduced in papers and retrieve information like price, features, and specs from the web.  
16. Detect and list contradictory findings from across the corpus.  
17. Compare the strategy used in paper A vs. paper B.  
18. Compile a list of definitions for complex research terms in paper A that do not have any introduction to them.  
19. List papers by the years of publication and which journals/conferences they were published in.  
20. Check if any claims in my draft paragraph lack support in the retrieved sources.

**Data**

The data I will use will be research papers; for the sake of not actually compromising sources from my thesis, I will attempt to curate a list of 20 other research papers and perform a couple different parsing strategies to organize them into a structured set. Some parsing strategies to explore could be parsing the arXiv HTML pages, PDF parsing, or other legal web scraping options as explored in HW2. Once structured, this data can either be maintained in a vector store for RAG implementation or just a simple SQL table.

*Potential Tool Usage:* Every query is handled via tools: db.query (SQLite), vector.search (FAISS over chunks), pdf.snippet (short quotes with page numbers), and a web.search tool for arXiv/Papers With Code/venue pages. 

*Web feature:* Queries that require fresh info (new SOTA, deadlines, recent papers) will call web.search and return links and citations.

**Models**

The two models my team has decided to use via Hugging Face are

*Large model:* Llama-3.3-70B-Instruct 

*Small model:* Llama-3.1-8B-Instruct 

**Deliverables**

The project will also demonstrate the following requirements:

*Prompting techniques:* prompt chaining (planner, tools, synthesis), meta-prompting (policies), and self-reflection (coverage/citation checks).

*Security testing:* 5 prompts covering instruction override, SQL/tool injection, data exfiltration, and web-borne prompt injection; guarded by allow-listed queries, parameterized DB access, and a lightweight injection classifier.

*Final Result:* Colab notebook, video demo code walkthrough, slides, report, document and question datasets/CSVs, and meeting notes.


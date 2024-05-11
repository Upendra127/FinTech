# Visualizing Stocks
This repository contains the necessary files required for visualisation of stocks (given tickers) as inferred from documents downloaded from 10K filings

---

## How to Run?
1. Run deleter.py
    - Ensure that the static folder has no previously stored images
    - Install python version 3.12.2
2. Add your HuggingFace Authorization key in parser.py (line no. 30)
3. Run app.py
    - Load the URL printed in the terminal logs on your browser
    - Enter number of companes (1-3) along with their ticker symbols and company names

Sample Instance Demo - [link](https://drive.google.com/file/d/1hWngrstO7b-yrEOqkZryqyJYrrUKAMZe/view) 

---

## Implementation Procedure - 
- Downloaded 10K filings of input ticker (from 1995 through 2023) using the sec_edgar_downloader python package
- The above downloaded HTML files are then parsed using parser.py to extract management comments (item 7)
- Extracted text is then fed to a BART model (loaded from HuggingFace API) to provide text summaries for each document.
- Text summaries act as input to a BERT model. For every input, the embedding of the [CLS] token is used as the summary embedding. 
    - Each summary embedding is compared against the embedding of a perfectly growing hypothetical company's summarized text (manually generated).
    - The above generated similarity score representing the growth of a company as described by one summarized text.
    - These scores are then visualised across years (as shown by the graph)
---

## What insights do we see?

Every similarity score, similarity of the summarized text with a perfectly growing hypothetical company's summarized text, (as described in procedure section) provides an estimate to how well the company progressed in the current year. Thus, putting together such estimates for each year gives us information of how well the company performed per year. 
Note: A decreasing trend need not necessarily indicate that the company declined rather it means that the company did not grow as much as it did in the previous year.

---

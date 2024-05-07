from flask import Flask, render_template, request, redirect, url_for
from sec_edgar_downloader import Downloader
from Sec_downloader import download_10k_filings
from tqdm import tqdm
import subprocess
from sorter import sorter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    # Get the number of companies from the form
    count = 0
    num_companies = int(request.form['numCompanies'])
    print(num_companies)
    # Create a list to store the company data
    companies = []
    tickers = []
    # Iterate over the form data and retrieve ticker symbols and company names
    for i in range(1, num_companies + 1):
        ticker = request.form[f'ticker{i}']
        company_name = request.form[f'company{i}']
        companies.append((ticker, company_name))
        tickers.append(ticker)
    # Download 10-K filings for the list of companies
    print("------Downloading-------")
    download_10k_filings(companies)
    for ticker in tickers:
        if(count > 0):
            download_10k_filings(companies)
        print(ticker)
        sorter(ticker)
        subprocess.run(["python", "parser.py"])
        count +=1

    # Redirect to a success page or back to the form page
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

from sec_edgar_downloader import Downloader
from tqdm import tqdm

def download_10k_filings(companies):
    for ticker, company_name in tqdm(companies):
        print(f"Downloading 10-K filings for {company_name} ({ticker})...")
        try:
            downloader = Downloader(company_name,email_address="Upendrakatara8@gmail.com")
            downloader.get("10-K", ticker,download_details=True ,after="2000-01-01", before="2024-03-25")  # Downloads the latest 5 filings
            print(f"Successfully downloaded 10-K filings for {company_name} ({ticker})")
        except Exception as e:
            print(f"Error downloading 10-K filings for {company_name} ({ticker}): {e}")

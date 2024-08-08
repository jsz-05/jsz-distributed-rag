import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

base_url = 'https://kmchandy.github.io/'

# Main page containing the links to subpages
main_page_url = base_url + 'table_of_contents.html'
excluded_urls = {
    'https://kmchandy.github.io/./index.html',
    'https://kmchandy.github.io/table_of_contents.html',
    'https://kmchandy.github.io/cross_reference.html',
    'https://kmchandy.github.io/index.html'
}

def fetch_html(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def format_text(content):
    # Remove extra spaces and newlines, keep paragraphs separated
    content = re.sub(r'\n+', '\n', content)  # Replace multiple newlines with a single newline    
    return content.strip()

def insert_figures_in_text(soup, page_url):
    # Find all figure elements
    figures = soup.find_all('figure')
    figure_map = {}
    
    for figure in figures:
        img = figure.find('img', src=True)
        caption = figure.find('figcaption')
        
        if img and caption:
            img_url = urljoin(page_url, img['src'])
            caption_text = caption.get_text(strip=True)
            figure_map[figure] = (img_url, caption_text)
    
    return figure_map

main_page_html = fetch_html(main_page_url)
soup = BeautifulSoup(main_page_html, 'html.parser')

links = soup.find_all('a', href=True)

subpage_urls = [urljoin(base_url, link['href']) for link in links if link['href'].endswith('.html')]

with open('corpus/new_reference.txt', 'w', encoding='utf-8') as file:
    # Iterate through each subpage URL
    for url in subpage_urls:
        if url in excluded_urls:
            continue

        try:
            # Fetch and parse the subpage HTML
            subpage_html = fetch_html(url)
            subpage_soup = BeautifulSoup(subpage_html, 'html.parser')
            
            # Extract figure details
            figure_map = insert_figures_in_text(subpage_soup, url)
            
            # Extract the title of the page
            title = subpage_soup.title.string if subpage_soup.title else 'No Title'
            
            # Extract and format the text content
            paragraphs = subpage_soup.find_all('p')
            formatted_content = []
            
            for para in paragraphs:
                para_text = para.get_text(strip=True)
                formatted_content.append(format_text(para_text))
                
                # Check for figures within paragraphs
                figures_in_para = para.find_all_next('figure', limit=1)
                for figure in figures_in_para:
                    if figure in figure_map:
                        img_url, caption = figure_map[figure]
                        formatted_content.append(f"\n\nFigure Link: {img_url}")
                        formatted_content.append(f"Caption: {caption}\n")
            
            # Join all paragraphs with figures
            content_with_figures = '\n\n'.join(formatted_content)
            
            # Write to the file
            file.write(f"--- START Content from {url} ---\n")
            file.write(f"TITLE: {title}\n\n")
            file.write(content_with_figures)
            file.write(f"\n--- END Content from {url} ---\n")
            file.write("\n\n")
            
            print(f"Scraped content from: {url}")

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")

print("Scraping completed.")

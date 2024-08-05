import requests
from bs4 import BeautifulSoup
import re

base_url = 'https://kmchandy.github.io/'

# main page containing the links to subpages
main_page_url = base_url + 'table_of_contents.html'
excluded_urls = {
    'https://kmchandy.github.io/./index.html',
    'https://kmchandy.github.io/./table_of_contents.html',
    'https://kmchandy.github.io/./cross_reference.html',
    'https://kmchandy.github.io/index.html'
}

def fetch_html(url):
    response = requests.get(url)
    response.raise_for_status() 
    return response.text

def format_text(content):
    # Replace newlines with spaces unless there is a blank line separating sections
    lines = content.split('\n')
    formatted_lines = []
    blank_line = False
    
    for line in lines:
        if line.strip() == '':
            # Preserve blank lines
            if formatted_lines and not blank_line:
                formatted_lines.append('')
            blank_line = True
        else:
            # Replace newline with space
            if blank_line:
                formatted_lines.append(' ')
            formatted_lines.append(line.strip())
            blank_line = False
    
    return ' '.join(formatted_lines)

main_page_html = fetch_html(main_page_url)
soup = BeautifulSoup(main_page_html, 'html.parser')

links = soup.find_all('a', href=True)

subpage_urls = [base_url + link['href'] for link in links if link['href'].endswith('.html')]

with open('reference.txt', 'w', encoding='utf-8') as file:
    # Iterate through each subpage URL
    for url in subpage_urls:
        if url in excluded_urls:
            continue

        try:
            # Fetch and parse the subpage HTML
            subpage_html = fetch_html(url)
            subpage_soup = BeautifulSoup(subpage_html, 'html.parser')
            
            # Extract the title of the page
            title = subpage_soup.title.string if subpage_soup.title else 'No Title'

            # Extract and format the text content
            text_content = subpage_soup.get_text(separator='\n', strip=True)
            formatted_content = format_text(text_content)
            
            # Write to the file
            file.write(f"--- Content from {url} ---\n")
            file.write(f"TITLE: {title}\n\n")
            file.write(formatted_content)
            file.write("\n\n")
            
            print(f"Scraped content from: {url}")

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")

print("Scraping completed.")


def replace_spaces_with_newlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    modified_content = content.replace('    ', '\n\n')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

file_path = 'corpus/reference.txt'

# replace_spaces_with_newlines(file_path)


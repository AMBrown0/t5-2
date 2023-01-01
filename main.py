# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pypdf import PdfReader, PageObject
import re
import requests
from bs4 import BeautifulSoup

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_web_page_text(url):

    # Set the URL of the website you want to read
    #url = "https://www.example.com"

    # Send an HTTP request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the website
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the text from the website
        text = soup.get_text()
    else:
        print("Failed to fetch the website")
    return text

def read_pdf(filename):
    all_text = ""
    reader = PdfReader(filename)

    num_pages = len(reader.pages)
    print(reader.pages[0])
    print(f"num_pages {num_pages}")
    num_pages=10
    for page in range(num_pages):
        text = reader.pages[page].extract_text()
        print(text)
        all_text += text

    all_text = re.sub(r'\\[a-z]+', '', all_text)  # Remove Latex commands
    all_text = re.sub(r'[^\w\s]', '', all_text)  # Remove punctuation

    return all_text

def write_text_to_file(filename,text):
    with open(filename, 'w') as f:
        f.write(text)
    f.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import torch
    import json
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    text = """
    The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.

    The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.

    At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.

    "We'll be the comeback kids, all of us," he said. "We want to get our country back."

    The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
    """
    text=read_pdf("./basic_statistics_for_forecasting.pdf")
    text=read_web_page_text(r"https://www.analyticsvidhya.com/blog/2022/01/introduction-to-knn-algorithms/#:~:text=KNN%20also%20called%20K%2D%20nearest,assumptions%20for%20underlying%20data%20assumptions.")
    write_text_to_file("./origonal_text.txt", text)
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    print("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=8000,
                                 early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n\nSummarized text: \n", output)
    write_text_to_file("./processed_text.txt", output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
import requests
import re
import os
import shutil
import time
import math

# This crawler fetches images aircraft images from airliners.net
# Program starts at crawl_airliners() given a num_images, and will go to the search
# page and pulls 84 (or some other specified number) images per page

# Requires a header to fool airliners.net into thinking that the program is a browser
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

# Function finds urls for all images on the search page
def find_page_urls(raw_page):
    base_url = 'https://www.airliners.net/'
    image_url_extensions = re.findall(r'results-photo">\n[\ ]*<a href="/(photo/[0-9a-z-?=%/]*)"', raw_page, flags=re.IGNORECASE)
    page_image_urls = [base_url + image_url for image_url in image_url_extensions]
    return page_image_urls

# Downloads the image of a particular image page
def download_image(img_url):
    # Fetch the raw page contents and scans for the filename
    img_res = requests.get(img_url, stream=True)
    filename = re.findall(r'/([0-9a-z.]+jpg)', img_url)
    
    # If there's no match on the page, skip/return
    if len(filename) == 0:
        return
    filename = filename[0]

    # Create "planes_images" directory if it does not exist
    if not os.path.isdir('./planes_images'):
        os.mkdir('./planes_images/')

    # Saves image to planes_images folder
    with open('./planes_images/' + filename, 'wb') as img_file:
        shutil.copyfileobj(img_res.raw, img_file)

# Loops through all image urls of a page and downloads every single image
def download_single_page(img_urls):
    for i, img_url in enumerate(img_urls):
        # Get raw contents of an image page and search for the image URL
        image_page_raw = requests.get(img_url, headers=header).text
        all_image_urls = re.findall(r'<img src="(https://imgproc.airliners.net/photos/[0-9a-z-?=%/.]*)', image_page_raw)
        # The third url of our regex should be the image URL
        # If there aren't enough matches, skip this image and continue to the next
        if len(all_image_urls) < 3:
            print(f'\tFailed to download image {i}/{84} of the page')
            continue
        image_url = all_image_urls[2]

        # Download the requested image
        download_image(image_url)
        print(f'\tDownloaded image {i}/{84} of the page')

# Start of program which fetches some number of pages depending on the num_images request
def crawl_airliners(num_images=100000, page_start=667, per_page=84):
    # Calculate how many pages should be fetched to get at least the requested number of images
    num_page_iters = math.ceil(num_images/per_page)
    
    # Loop through the pages, fetch their image page URLs and downlaod them
    url = 'https://www.airliners.net/search?perPage=' + str(per_page) + '&page='
    for i in range(page_start, num_page_iters + page_start):

        print(f'Fetching page {i}/{num_page_iters + page_start}')
        curr_page_url = url + str(i)
        curr_page_text = requests.get(curr_page_url, headers=header).text
        page_img_urls = find_page_urls(curr_page_text)
        download_single_page(page_img_urls)
    print('\tDone fetching page')

num_imgs, page_start = 100000, 770
done = False
while not done:
    try:
        crawl_airliners(num_imgs, page_start)
        done = True
    except:
        page_start += 100
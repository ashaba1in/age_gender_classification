from urllib import request

import tqdm
from selenium import webdriver

browser = webdriver.Firefox()
browser.get("https://unsplash.com/s/photos/sculpture")

images = browser.find_elements_by_tag_name('img')
sources = []
for image in images:
    sources.append(image.get_attribute('src'))

browser.close()
sources = sources[2:]
print(len(sources))

for file_name, source in tqdm.tqdm(enumerate(sources), disable=True):
    with request.urlopen(source) as response, open('sculptures/' + str(file_name) + '.jpeg', 'wb') as out_file:
        data = response.read()
        out_file.write(data)

from urllib import request

import tqdm
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


def find(driver):
    element = driver.find_elements_by_xpath('//*[@title="Download photo"]')
    if element:
        return element
    else:
        return False


browser = webdriver.Firefox()
browser.get("https://unsplash.com/s/photos/sculpture")

sources = set([])
shift = 5000
pos = 0

while len(sources) < 100:
    browser.execute_script("window.scrollTo(0, {})".format(shift * pos))
    pos += 1
    images = WebDriverWait(browser, 20).until(find)
    print('Нашел {} картинок, всего уже {}'.format(len(images), len(sources)))
    for image in images:
        try:
            sources.add(image.get_attribute('href'))
        except Exception as e:
            print(e.__class__)

browser.close()
print(sources)
print(len(sources))

for file_name, source in tqdm.tqdm(enumerate(sources), disable=True):
    with request.urlopen(source) as response, open('../sculptures/' + str(file_name), 'wb') as out_file:
        data = response.read()
        out_file.write(data)

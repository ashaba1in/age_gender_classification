import argparse
from urllib import request

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def find(driver):
    element = driver.find_elements_by_xpath('//*[@title="Download photo"]')
    if element:
        return element
    else:
        return False


def get_argv():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--load_url', type=str)
    parser.add_argument('--load_count', type=int)
    
    argv = parser.parse_args()
    return argv


def main():
    browser = webdriver.Firefox()
    browser.get(get_argv().load_url)
    
    sources = set([])
    shift = 1000
    pos = 0
    
    while len(sources) < get_argv().load_count:
        browser.execute_script("window.scrollTo(0, {})".format(shift * pos))
        pos += 1
        images = WebDriverWait(browser, 100).until(find)
        print('Нашел {} картинок, всего уже {}'.format(len(images), len(sources)))
        for image in images:
            try:
                sources.add(image.get_attribute('href'))
            except Exception as e:
                print(e.__class__)
    
    browser.close()

    for file_name, source in tqdm(enumerate(sources), disable=True):
        with request.urlopen(source) as response, open(get_argv().save_path + str(file_name), 'wb') as out_file:
            data = response.read()
            out_file.write(data)


if __name__ == '__main__':
    main()

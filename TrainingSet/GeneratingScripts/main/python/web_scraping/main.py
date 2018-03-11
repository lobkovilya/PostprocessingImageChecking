import requests
import re
import uuid
from lxml import html
from os import path


def read_file(filename):
    with open(filename) as input_file:
        text = input_file.read()
    return text

def download_image_from_page(url, folder):
    try:
        r = requests.get(url)
        tree = html.fromstring(r.text)

        img_link = tree.xpath('//div[@id="allsizes-photo"]/descendant::img')[0].get('src')
        img_content = requests.get(img_link).content
        img_name = str(uuid.uuid4()) + ".jpg";

        with open(path.join(folder, img_name), 'wb') as image_file:
            image_file.write(img_content)

        print('download image ' + img_name)
    except:
        print('fail')

def download_images_from_url(url, folder):
    print('Start with url: ' + url)

    p = re.compile('\D*www.flickr.com/photos/\D*');

    r = requests.get(url)
    tree = html.fromstring(r.text)
    links = tree.xpath('//a')

    links = [l.get('href') for l in links]
    links = list(filter(lambda x: p.match(x), links))
    links = list(map(lambda x: x[:-2] + 'k/', links))

    for l in links:
        download_image_from_page(l, folder)

    print('Done with url: ' + url)

if __name__ == '__main__':
    # url = 'https://pixelpeeper.com/cameras/?camera=2104'
    template = 'https://pixelpeeper.com/cameras/?camera=1451&perpage=25&iso_min=none&iso_max=none&exp_min=none&exp_max=none&res=3&digicam=0'

    urls = []
    for i in range(1, 10):
        urls.append(template + '&p=' + str(i))

    # print(urls)
    for url in urls:
        download_images_from_url(url, './images/')
    # download_images_from_url(url, './images/')

        # print(html.tostring(l))
        # print(links)
        # with open('test.html', 'wb') as output_file:
        # output_file.write(r.text.encode('cp1251'))